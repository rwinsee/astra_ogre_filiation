#!/usr/bin/env python3
# %%
# =============================================================================
# OGRE — GÉNÉRATION COMPLÈTE DES PRÉCONISATIONS DE FAMILLES OGR
# =============================================================================
#
# Ce programme produit, à partir du référentiel des appellations ROME 4.61 :
#
#   1. ogre_familles_fortes.parquet
#   2. ogre_familles_plausibles.parquet
#   3. ogre_rapprochements_a_arbitrer.parquet
#
# Il est organisé en cellules « # %% » :
#   - chaque bloc peut être lancé séparément dans VSCode ;
#   - le fichier peut également être exécuté intégralement dans un terminal.
#
# Exemple dans le service VSCode Python de SSPCloud :
#
#   cd ~/work/astra_ogre_filiation
#   /opt/python/bin/python notebooks/generation_preconisations_ogre.py
#
# Le programme réalise réellement les calculs suivants :
#   - nettoyage et lemmatisation des libellés ;
#   - recherche de voisins par le sens, l'écriture et le vocabulaire ;
#   - calcul de quatre similarités ;
#   - ajout des bonus de contexte ;
#   - détection des divergences métier ;
#   - calcul du score composite et du score de confiance ;
#   - classement en FORTE, PLAUSIBLE ou A_ARBITRER ;
#   - choix d'un parent proposé ;
#   - constitution des familles ;
#   - contrôles et export en Parquet ;
#   - envoi facultatif des fichiers vers S3.
#
# IMPORTANT
# ---------
# Les scores sont des indicateurs heuristiques destinés à classer des pistes.
# Ils ne constituent pas des probabilités et ne remplacent pas l'arbitrage des
# agents. Les seuils de cette V1 doivent être audités sur un échantillon métier.
# =============================================================================


# %%
# =============================================================================
# ÉTAPE 0 — IMPORTS
# =============================================================================

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import pyarrow
import requests
import s3fs
import sklearn
import spacy
import wn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# %%
# =============================================================================
# ÉTAPE 1 — PARAMÈTRES GÉNÉRAUX
# =============================================================================
#
# Les paramètres fonctionnels sont regroupés ici. Pour créer une V2, il est
# conseillé de modifier VERSION_METHODE plutôt que d'écraser silencieusement
# une production V1.
# =============================================================================

VERSION_ROME = "4.61"
VERSION_ROME_FICHIER = "v461"
VERSION_METHODE = "OGRE_RAPPROCHEMENT_V1"

URL_FRANCE_TRAVAIL = (
    "https://api.francetravail.fr/"
    "api-nomenclatureemploi/v1/open-data/csv"
)

MODELE_SPACY = "fr_core_news_md"
WOLF_SPEC = "omw-fr:2.0"

# Le fichier se trouve normalement dans notebooks/. Le petit mécanisme
# ci-dessous permet aussi de lancer une copie placée temporairement à la racine.
CHEMIN_SCRIPT = Path(__file__).resolve()
if CHEMIN_SCRIPT.parent.name == "notebooks":
    RACINE_PROJET = CHEMIN_SCRIPT.parents[1]
else:
    RACINE_PROJET = CHEMIN_SCRIPT.parent

DOSSIER_SOURCE = RACINE_PROJET / "data" / "raw"
DOSSIER_INTERMEDIAIRE = RACINE_PROJET / "data" / "intermediate"
DOSSIER_SORTIE = RACINE_PROJET / "data" / "output"

NOM_FICHIER_SOURCE = "unix_referentiel_appellation_v461_utf8.csv"
CHEMIN_SOURCE = DOSSIER_SOURCE / NOM_FICHIER_SOURCE

PREFIXE_CACHE = "v461_ogre_rapprochement_v1"
CHEMIN_APPELLATIONS_PREPAREES = (
    DOSSIER_INTERMEDIAIRE / f"{PREFIXE_CACHE}_appellations_preparees.parquet"
)
CHEMIN_VECTEURS_SEMANTIQUES = (
    DOSSIER_INTERMEDIAIRE / f"{PREFIXE_CACHE}_vecteurs_semantiques.npy"
)
CHEMIN_PAIRES_SCOREES = (
    DOSSIER_INTERMEDIAIRE / f"{PREFIXE_CACHE}_paires_scorees.parquet"
)

CHEMIN_SORTIE_FORTE = DOSSIER_SORTIE / "ogre_familles_fortes.parquet"
CHEMIN_SORTIE_PLAUSIBLE = DOSSIER_SORTIE / "ogre_familles_plausibles.parquet"
CHEMIN_SORTIE_ARBITRER = (
    DOSSIER_SORTIE / "ogre_rapprochements_a_arbitrer.parquet"
)
CHEMIN_MANIFESTE = DOSSIER_SORTIE / "ogre_preconisations_manifeste.json"

# Chemin cible SSPCloud/MinIO. « travail » est le bucket ; le reste est le
# préfixe dans ce bucket.
DOSSIER_S3 = (
    "s3://travail/projet-edep-artemis/open_data/"
    "astra_ogre/referentiels/preconisations"
)

# Contrôles attendus pour le référentiel ROME 4.61.
NB_OGR_ATTENDU = 14_301
NB_ROME_ATTENDU = 1_911
NB_PRINCIPALES_ATTENDU = 1_911
NB_SYNONYMES_ATTENDU = 12_390

# Nombre de voisins retenus par porte d'entrée.
NB_VOISINS_SEMANTIQUES = 12
NB_VOISINS_CARACTERES = 12
NB_VOISINS_LEXICAUX = 10

# Taille des paquets pour limiter l'occupation mémoire.
TAILLE_BATCH_SPACY = 256
TAILLE_BATCH_SCORES = 50_000


def lire_booleen_env(nom: str, valeur_par_defaut: bool) -> bool:
    """Lit proprement un booléen transmis par variable d'environnement."""

    valeur = os.getenv(nom)
    if valeur is None:
        return valeur_par_defaut
    return valeur.strip().lower() in {"1", "true", "vrai", "yes", "oui"}


# False : le CSV déjà présent est réutilisé.
FORCER_TELECHARGEMENT = lire_booleen_env(
    "OGRE_FORCER_TELECHARGEMENT",
    False,
)

# True : ignore les caches intermédiaires et refait tous les calculs lourds.
FORCER_RECALCUL_COMPLET = lire_booleen_env(
    "OGRE_FORCER_RECALCUL_COMPLET",
    False,
)

# Le modèle spaCy et WOLF peuvent être téléchargés lors de la première
# exécution. Les exécutions suivantes utilisent les ressources locales.
TELECHARGER_MODELE_SPACY_SI_ABSENT = lire_booleen_env(
    "OGRE_TELECHARGER_SPACY",
    True,
)
UTILISER_WOLF = lire_booleen_env("OGRE_UTILISER_WOLF", True)
TELECHARGER_WOLF_SI_ABSENT = lire_booleen_env(
    "OGRE_TELECHARGER_WOLF",
    True,
)

# Par sécurité, l'envoi S3 n'est pas déclenché par défaut. Après contrôle des
# sorties locales, lancer par exemple :
#
#   OGRE_UPLOAD_S3=1 /opt/python/bin/python \
#     notebooks/generation_preconisations_ogre.py
ENVOYER_VERS_S3 = lire_booleen_env("OGRE_UPLOAD_S3", False)

for dossier in (DOSSIER_SOURCE, DOSSIER_INTERMEDIAIRE, DOSSIER_SORTIE):
    dossier.mkdir(parents=True, exist_ok=True)


# %%
# =============================================================================
# ÉTAPE 2 — POIDS ET SEUILS DE LA MÉTHODE V1
# =============================================================================
#
# Les poids ne proviennent pas d'un modèle statistique entraîné : il n'existe
# pas encore de jeu de décisions OGRE suffisamment important et stabilisé pour
# apprendre ces poids. Ils traduisent donc une hiérarchie explicite :
#
#   1. le sens général du libellé est prioritaire ;
#   2. la proximité d'écriture apporte un indice fort ;
#   3. le vocabulaire métier explicite complète le sens ;
#   4. le recouvrement brut sert d'indice de confirmation.
#
# Les bonus restent volontairement faibles : appartenir au même ROME ou avoir
# un terme commun ne suffit jamais, à lui seul, à créer une famille.
# =============================================================================

POIDS_SEMANTIQUE = 0.43
POIDS_CARACTERES = 0.29
POIDS_LEXICAL = 0.20
POIDS_JACCARD = 0.08

BONUS_MEME_ROME = 0.035
BONUS_MEME_TERME_INITIAL = 0.025
BONUS_SYNONYMIE_WOLF = 0.040

# Seuils utilisés par les règles détaillées de classement plus bas.
SEUIL_FORTE_MEME_ROME = 0.84
SEUIL_PLAUSIBLE_MEME_ROME = 0.68
SEUIL_PLAUSIBLE_INTER_ROME = 0.80
SEUIL_ARBITRER_STANDARD = 0.60
SEUIL_ARBITRER_INTER_ROME = 0.56


# %%
# =============================================================================
# ÉTAPE 3 — VOCABULAIRE DE CONTRÔLE MÉTIER
# =============================================================================
#
# Ces listes servent uniquement de garde-fou. Par exemple, « directeur » et
# « assistant directeur » peuvent être proches textuellement, mais ne doivent
# pas être classés automatiquement au niveau fort.
# =============================================================================

MARQUEURS_FONCTION = {
    "adjoint",
    "aide",
    "apprenti",
    "assistant",
    "chef",
    "coordinateur",
    "directeur",
    "dirigeant",
    "ingenieur",
    "junior",
    "manager",
    "operateur",
    "responsable",
    "senior",
    "stagiaire",
    "superviseur",
    "technicien",
}

MARQUEURS_SUBORDONNES = {
    "adjoint",
    "aide",
    "apprenti",
    "assistant",
    "junior",
    "stagiaire",
}

# Certains mots sont habituellement considérés comme des mots outils par les
# traitements linguistiques, mais doivent être conservés ici.
MOTS_OUTILS_A_CONSERVER = {
    "avec",
    "sans",
}


# %%
# =============================================================================
# ÉTAPE 4 — OUTILS GÉNÉRAUX
# =============================================================================


def afficher_titre(numero: int, titre: str) -> None:
    """Affiche un séparateur lisible dans le terminal."""

    print("\n" + "=" * 78)
    print(f"ÉTAPE {numero} — {titre}")
    print("=" * 78)


def afficher_duree(debut: float, libelle: str) -> None:
    """Affiche la durée écoulée depuis un time.perf_counter()."""

    duree = time.perf_counter() - debut
    print(f"{libelle} : {duree:,.1f} secondes")


def identifiant_stable(*valeurs: object, longueur: int = 20) -> str:
    """Construit un identifiant reproductible à partir de valeurs métier."""

    contenu = "|".join("" if valeur is None else str(valeur) for valeur in valeurs)
    return hashlib.sha256(contenu.encode("utf-8")).hexdigest()[:longueur]


def sans_accents(texte: str) -> str:
    """Retire les accents après décomposition Unicode."""

    decompose = unicodedata.normalize("NFKD", str(texte))
    return "".join(car for car in decompose if not unicodedata.combining(car))


def prefixe_commun(a: str, b: str) -> int:
    """Retourne la longueur du préfixe commun de deux mots."""

    compteur = 0
    for caractere_a, caractere_b in zip(a, b):
        if caractere_a != caractere_b:
            break
        compteur += 1
    return compteur


def neutraliser_variante_genre(texte: str) -> str:
    """
    Neutralise prudemment une variante de genre écrite avec « / ».

    Exemple :
        « Dompteur / Dompteuse d'animaux »
        devient approximativement
        « Dompteur d'animaux ».

    La neutralisation n'est faite que lorsque les deux mots entourant la barre
    oblique possèdent un préfixe suffisamment proche. Une barre oblique qui
    sépare deux notions différentes est simplement transformée en espace.
    """

    resultat = unicodedata.normalize("NFKC", str(texte))
    motif = re.compile(
        r"(?P<gauche>[A-Za-zÀ-ÖØ-öø-ÿŒœÆæ-]+)"
        r"\s*/\s*"
        r"(?P<droite>[A-Za-zÀ-ÖØ-öø-ÿŒœÆæ-]+)"
    )

    def remplacer(correspondance: re.Match[str]) -> str:
        gauche = sans_accents(correspondance.group("gauche").lower())
        droite = sans_accents(correspondance.group("droite").lower())

        # Trois lettres communes couvrent notamment acteur/actrice,
        # conducteur/conductrice et infirmier/infirmière.
        if prefixe_commun(gauche, droite) >= 3:
            return correspondance.group("gauche")

        # Sinon, on ne suppose pas qu'il s'agit d'une variation de genre.
        return (
            correspondance.group("gauche")
            + " "
            + correspondance.group("droite")
        )

    precedent = None
    while precedent != resultat:
        precedent = resultat
        resultat = motif.sub(remplacer, resultat)

    return resultat.replace("/", " ")


def normaliser_identite(texte: str) -> str:
    """
    Produit une forme stable pour comparer deux écritures.

    Les accents, la casse, les apostrophes, les tirets et la ponctuation sont
    neutralisés. Les nombres et les mots « avec »/« sans » sont conservés.
    """

    texte = neutraliser_variante_genre(texte)
    texte = texte.replace("’", "'").replace("`", "'")
    texte = sans_accents(texte).lower()
    texte = re.sub(r"[^a-z0-9]+", " ", texte)
    return re.sub(r"\s+", " ", texte).strip()


def est_peu_usite(valeur: object) -> bool:
    """Interprète les différentes écritures possibles de peu_usite."""

    normalisee = sans_accents(str(valeur)).strip().lower()
    return normalisee in {
        "1",
        "o",
        "oui",
        "true",
        "vrai",
        "x",
        "peu usite",
    }


def joindre_ensemble(valeurs: set[str]) -> str:
    """Stocke un petit ensemble sous une forme stable compatible Parquet."""

    return "|".join(sorted(valeur for valeur in valeurs if valeur))


def lire_ensemble(valeur: object) -> set[str]:
    """Reconstruit un ensemble précédemment stocké avec joindre_ensemble."""

    texte = str(valeur).strip()
    if not texte:
        return set()
    return {element for element in texte.split("|") if element}


# Contrôles simples des fonctions de normalisation.
assert normaliser_identite("Dompteur / Dompteuse") == "dompteur"
assert normaliser_identite("Monteur-câbleur") == "monteur cableur"


# %%
# =============================================================================
# ÉTAPE 5 — CONTRÔLE DE L'ENVIRONNEMENT
# =============================================================================

afficher_titre(5, "CONTRÔLE DE L'ENVIRONNEMENT")

print("Python              :", sys.version.splitlines()[0])
print("Dossier de travail  :", Path.cwd())
print("Racine du projet    :", RACINE_PROJET)
print("pandas              :", pd.__version__)
print("numpy               :", np.__version__)
print("pyarrow             :", pyarrow.__version__)
print("scikit-learn        :", sklearn.__version__)
print("spaCy               :", spacy.__version__)
print("boto3               :", boto3.__version__)
print("Méthode             :", VERSION_METHODE)
print("Recalcul complet     :", FORCER_RECALCUL_COMPLET)
print("Envoi S3             :", ENVOYER_VERS_S3)


# %%
# =============================================================================
# ÉTAPE 6 — TÉLÉCHARGEMENT OU RÉUTILISATION DU RÉFÉRENTIEL
# =============================================================================

afficher_titre(6, "RÉCUPÉRATION DU RÉFÉRENTIEL FRANCE TRAVAIL")

if CHEMIN_SOURCE.exists() and not FORCER_TELECHARGEMENT:
    chemin_csv = CHEMIN_SOURCE
    print("Référentiel local réutilisé :", chemin_csv)
else:
    print("Téléchargement depuis :", URL_FRANCE_TRAVAIL)
    reponse = requests.get(URL_FRANCE_TRAVAIL, timeout=300)
    reponse.raise_for_status()
    contenu = reponse.content

    if zipfile.is_zipfile(io.BytesIO(contenu)):
        with zipfile.ZipFile(io.BytesIO(contenu)) as archive:
            candidats = [
                nom
                for nom in archive.namelist()
                if "referentiel_appellation" in nom.lower()
                and nom.lower().endswith(".csv")
                and VERSION_ROME_FICHIER in nom.lower()
            ]
            if not candidats:
                raise FileNotFoundError(
                    "Le CSV des appellations ROME 4.61 n'a pas été trouvé "
                    f"dans l'archive. Contenu : {archive.namelist()}"
                )

            nom_dans_archive = sorted(candidats)[0]
            with archive.open(nom_dans_archive) as source:
                with CHEMIN_SOURCE.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
    else:
        CHEMIN_SOURCE.write_bytes(contenu)

    chemin_csv = CHEMIN_SOURCE
    print("Référentiel téléchargé :", chemin_csv)

if not chemin_csv.exists() or chemin_csv.stat().st_size == 0:
    raise FileNotFoundError(f"Référentiel absent ou vide : {chemin_csv}")

print(
    "Taille du fichier       :",
    round(chemin_csv.stat().st_size / 1_000_000, 2),
    "Mo",
)


# %%
# =============================================================================
# ÉTAPE 7 — LECTURE ET CONTRÔLE DU RÉFÉRENTIEL
# =============================================================================

afficher_titre(7, "LECTURE ET CONTRÔLE DU RÉFÉRENTIEL")

appellations_source = pd.read_csv(
    chemin_csv,
    sep=None,
    engine="python",
    encoding="utf-8-sig",
    dtype=str,
    keep_default_na=False,
)

appellations_source.columns = [
    str(colonne).replace("\ufeff", "").strip().lower()
    for colonne in appellations_source.columns
]

colonnes_obligatoires = {
    "code_ogr",
    "libelle_appellation_long",
    "code_rome",
    "classification",
}

colonnes_absentes = colonnes_obligatoires - set(appellations_source.columns)
if colonnes_absentes:
    raise ValueError(
        "Colonnes absentes du référentiel : "
        f"{sorted(colonnes_absentes)}"
    )

# peu_usite est utile pour le choix du parent, mais son absence ne doit pas
# empêcher le calcul. Une valeur vide équivaut alors à « non signalé ».
if "peu_usite" not in appellations_source.columns:
    appellations_source["peu_usite"] = ""
    print("Information : colonne peu_usite absente, valeur vide utilisée.")

colonnes_texte = [
    "code_ogr",
    "code_rome",
    "libelle_appellation_long",
    "classification",
    "peu_usite",
]

for colonne in colonnes_texte:
    appellations_source[colonne] = (
        appellations_source[colonne].astype(str).str.strip()
    )

appellations_source["classification"] = (
    appellations_source["classification"].str.upper()
)

# L'ordre par code OGR garantit des identifiants et des calculs reproductibles.
appellations_source = (
    appellations_source
    .sort_values("code_ogr", kind="stable")
    .reset_index(drop=True)
)
appellations_source["index_ogr"] = np.arange(
    len(appellations_source),
    dtype=np.int32,
)

nb_lignes = len(appellations_source)
nb_ogr = appellations_source["code_ogr"].nunique()
nb_rome = appellations_source["code_rome"].nunique()
nb_principales = int(
    (appellations_source["classification"] == "PRINCIPALE").sum()
)
nb_synonymes = int(
    (appellations_source["classification"] == "SYNONYME").sum()
)
nb_doublons_ogr = int(appellations_source["code_ogr"].duplicated().sum())

print("Nombre de lignes          :", nb_lignes)
print("Codes OGR distincts       :", nb_ogr)
print("Codes ROME distincts      :", nb_rome)
print("Appellations principales  :", nb_principales)
print("Appellations synonymes    :", nb_synonymes)
print("Codes OGR en doublon      :", nb_doublons_ogr)

assert nb_lignes == NB_OGR_ATTENDU
assert nb_ogr == NB_OGR_ATTENDU
assert nb_rome == NB_ROME_ATTENDU
assert nb_principales == NB_PRINCIPALES_ATTENDU
assert nb_synonymes == NB_SYNONYMES_ATTENDU
assert nb_doublons_ogr == 0

print("Contrôle du référentiel ROME 4.61 : OK")


# %%
# =============================================================================
# ÉTAPE 8 — CHARGEMENT DU MODÈLE LINGUISTIQUE FRANÇAIS
# =============================================================================
#
# Cette étape est sautée lorsqu'un cache complet de préparation existe.
# Le modèle moyen « fr_core_news_md » est requis car, contrairement au petit
# modèle, il contient des vecteurs de mots utilisables pour la proximité par
# le sens.
# =============================================================================


def charger_modele_spacy() -> spacy.language.Language:
    """Charge fr_core_news_md et le télécharge si l'option l'autorise."""

    try:
        return spacy.load(MODELE_SPACY)
    except OSError as premiere_erreur:
        if not TELECHARGER_MODELE_SPACY_SI_ABSENT:
            raise RuntimeError(
                f"Le modèle spaCy {MODELE_SPACY} est absent. Installer avec :\n"
                f"/opt/python/bin/python -m spacy download {MODELE_SPACY}"
            ) from premiere_erreur

        print(f"Téléchargement initial du modèle spaCy {MODELE_SPACY}...")
        try:
            from spacy.cli import download

            download(MODELE_SPACY)
            return spacy.load(MODELE_SPACY)
        except Exception as seconde_erreur:
            raise RuntimeError(
                f"Impossible d'installer le modèle {MODELE_SPACY}. "
                "Exécuter manuellement :\n"
                f"/opt/python/bin/python -m spacy download {MODELE_SPACY}"
            ) from seconde_erreur


# %%
# =============================================================================
# ÉTAPE 9 — NORMALISATION, LEMMATISATION ET VECTEURS SÉMANTIQUES
# =============================================================================
#
# Un lemme est la forme de référence d'un mot : « installations » devient
# par exemple « installation ». Le texte lemmatisé sert à comparer le
# vocabulaire sans confondre une simple variation grammaticale avec un autre
# métier.
# =============================================================================

afficher_titre(9, "PRÉPARATION LINGUISTIQUE DES APPELLATIONS")

cache_preparation_disponible = (
    CHEMIN_APPELLATIONS_PREPAREES.exists()
    and CHEMIN_VECTEURS_SEMANTIQUES.exists()
    and not FORCER_RECALCUL_COMPLET
)

if cache_preparation_disponible:
    print("Réutilisation du cache :", CHEMIN_APPELLATIONS_PREPAREES)
    appellations = pd.read_parquet(CHEMIN_APPELLATIONS_PREPAREES)
    vecteurs_semantiques = np.load(CHEMIN_VECTEURS_SEMANTIQUES)

    # Sécurise l'alignement entre le cache et le CSV courant.
    if appellations["code_ogr"].tolist() != appellations_source["code_ogr"].tolist():
        raise RuntimeError(
            "Le cache de préparation ne correspond plus au référentiel. "
            "Relancer avec OGRE_FORCER_RECALCUL_COMPLET=1."
        )
else:
    debut_preparation = time.perf_counter()
    nlp = charger_modele_spacy()
    print("Modèle spaCy chargé     :", MODELE_SPACY)
    print("Dimension des vecteurs  :", nlp.vocab.vectors_length)

    if nlp.vocab.vectors_length <= 0:
        raise RuntimeError(
            f"Le modèle {MODELE_SPACY} ne fournit pas de vecteurs de mots."
        )

    appellations = appellations_source[
        [
            "index_ogr",
            "code_ogr",
            "libelle_appellation_long",
            "code_rome",
            "classification",
            "peu_usite",
        ]
    ].copy()

    textes_neutralises = [
        neutraliser_variante_genre(libelle)
        for libelle in appellations["libelle_appellation_long"]
    ]
    appellations["libelle_normalise"] = [
        normaliser_identite(libelle)
        for libelle in appellations["libelle_appellation_long"]
    ]

    dimension = int(nlp.vocab.vectors_length)
    vecteurs_semantiques = np.zeros(
        (len(appellations), dimension),
        dtype=np.float32,
    )

    textes_lemmatises: list[str] = []
    termes_initiaux: list[str] = []
    termes_initiaux_wolf: list[str] = []
    nombres_stockes: list[str] = []
    marqueurs_stockes: list[str] = []
    avec_sans_stockes: list[str] = []

    composants_a_desactiver = [
        nom
        for nom in ("parser", "ner", "senter")
        if nom in nlp.pipe_names
    ]

    flux_docs = nlp.pipe(
        textes_neutralises,
        batch_size=TAILLE_BATCH_SPACY,
        disable=composants_a_desactiver,
    )

    for position, doc in enumerate(flux_docs):
        lemmes_normalises: list[str] = []
        lemmes_wolf: list[str] = []
        vecteurs_tokens: list[np.ndarray] = []

        for token in doc:
            if token.is_space or token.is_punct:
                continue

            lemme_brut = (token.lemma_ or token.text).strip().lower()
            lemme_normalise = normaliser_identite(lemme_brut)
            if not lemme_normalise:
                continue

            conserver = (
                not token.is_stop
                or lemme_normalise in MOTS_OUTILS_A_CONSERVER
                or token.like_num
            )
            if not conserver:
                continue

            lemmes_normalises.append(lemme_normalise)
            lemmes_wolf.append(lemme_brut)

            if token.has_vector and token.vector_norm > 0:
                vecteurs_tokens.append(token.vector.astype(np.float32, copy=False))

        if not lemmes_normalises:
            # Repli rare : conserver au moins la forme normalisée du libellé.
            lemmes_normalises = appellations.at[
                position,
                "libelle_normalise",
            ].split()
            lemmes_wolf = lemmes_normalises.copy()

        textes_lemmatises.append(" ".join(lemmes_normalises))
        termes_initiaux.append(lemmes_normalises[0] if lemmes_normalises else "")
        termes_initiaux_wolf.append(lemmes_wolf[0] if lemmes_wolf else "")

        ensemble_lemmes = set(lemmes_normalises)
        nombres_stockes.append(
            joindre_ensemble(
                set(re.findall(r"\b\d+\b", appellations.at[position, "libelle_normalise"]))
            )
        )
        marqueurs_stockes.append(
            joindre_ensemble(ensemble_lemmes & MARQUEURS_FONCTION)
        )
        avec_sans_stockes.append(
            joindre_ensemble(ensemble_lemmes & {"avec", "sans"})
        )

        if vecteurs_tokens:
            vecteur = np.mean(vecteurs_tokens, axis=0, dtype=np.float32)
        elif doc.vector_norm > 0:
            vecteur = doc.vector.astype(np.float32, copy=False)
        else:
            vecteur = np.zeros(dimension, dtype=np.float32)

        norme = float(np.linalg.norm(vecteur))
        if norme > 0:
            vecteurs_semantiques[position] = vecteur / norme

        if (position + 1) % 1_000 == 0:
            print(f"Appellations préparées : {position + 1:,}/{len(appellations):,}")

    appellations["texte_lemmatise"] = textes_lemmatises
    appellations["terme_metier_initial"] = termes_initiaux
    appellations["terme_metier_initial_wolf"] = termes_initiaux_wolf
    appellations["nombres"] = nombres_stockes
    appellations["marqueurs_fonction"] = marqueurs_stockes
    appellations["marqueurs_avec_sans"] = avec_sans_stockes
    appellations["peu_usite_bool"] = (
        appellations["peu_usite"].map(est_peu_usite).astype(bool)
    )

    appellations.to_parquet(
        CHEMIN_APPELLATIONS_PREPAREES,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    np.save(CHEMIN_VECTEURS_SEMANTIQUES, vecteurs_semantiques)

    afficher_duree(debut_preparation, "Préparation linguistique terminée")
    print("Cache écrit :", CHEMIN_APPELLATIONS_PREPAREES)

assert len(appellations) == NB_OGR_ATTENDU
assert vecteurs_semantiques.shape[0] == NB_OGR_ATTENDU


# %%
# =============================================================================
# ÉTAPE 10 — INITIALISATION FACULTATIVE DE WOLF
# =============================================================================
#
# WOLF est un dictionnaire lexical français. Il sert uniquement à confirmer
# que deux premiers termes métier différents appartiennent à un même ensemble
# de synonymes. Si WOLF est indisponible, le calcul continue sans ce bonus et
# le manifeste final l'indique explicitement.
# =============================================================================


def initialiser_wolf() -> tuple[object | None, bool]:
    """Retourne l'objet WOLF et un indicateur d'activation."""

    if not UTILISER_WOLF:
        print("WOLF désactivé par le paramètre OGRE_UTILISER_WOLF.")
        return None, False

    try:
        ressource = wn.Wordnet(WOLF_SPEC)
        # La requête force un accès réel à la ressource.
        ressource.synsets("chat")
        print("Ressource WOLF utilisée  :", WOLF_SPEC)
        return ressource, True
    except Exception as premiere_erreur:
        if not TELECHARGER_WOLF_SI_ABSENT:
            print("WOLF absent : calcul poursuivi sans bonus de synonymie.")
            print("Détail :", premiere_erreur)
            return None, False

        print("Téléchargement initial de WOLF :", WOLF_SPEC)
        try:
            wn.download(WOLF_SPEC)
            ressource = wn.Wordnet(WOLF_SPEC)
            ressource.synsets("chat")
            print("Ressource WOLF installée :", WOLF_SPEC)
            return ressource, True
        except Exception as seconde_erreur:
            print("AVERTISSEMENT : WOLF n'a pas pu être installé.")
            print("Le traitement continue sans le bonus WOLF.")
            print("Détail :", seconde_erreur)
            return None, False


# %%
# =============================================================================
# ÉTAPE 11 — VECTORISATION TF-IDF
# =============================================================================
#
# Deux représentations complémentaires sont créées :
#
#   - caractères : suites de 3 à 5 caractères, robustes aux tirets et petites
#     variations d'écriture ;
#   - lexicale : mots et groupes de deux mots lemmatisés, qui représentent le
#     vocabulaire métier commun.
#
# TF-IDF donne davantage de poids aux éléments informatifs et moins de poids
# aux éléments présents dans une grande partie du référentiel.
# =============================================================================


def construire_matrices_tfidf(
    table: pd.DataFrame,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Construit les matrices caractères et lexicales normalisées en L2."""

    vectoriseur_caracteres = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        lowercase=False,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )

    vectoriseur_lexical = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        lowercase=False,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )

    matrice_caracteres = vectoriseur_caracteres.fit_transform(
        table["libelle_normalise"].fillna("")
    ).tocsr()
    matrice_lexicale = vectoriseur_lexical.fit_transform(
        table["texte_lemmatise"].fillna("")
    ).tocsr()

    print("Matrice caractères :", matrice_caracteres.shape)
    print("Matrice lexicale    :", matrice_lexicale.shape)

    return matrice_caracteres, matrice_lexicale


# %%
# =============================================================================
# ÉTAPE 12 — GÉNÉRATION DES PAIRES CANDIDATES
# =============================================================================
#
# Comparer exhaustivement 14 301 appellations représente 102 252 150 paires.
# On limite donc les calculs détaillés aux paires trouvées par au moins une des
# quatre portes d'entrée documentées.
# =============================================================================

SOURCE_MEME_ROME = 1
SOURCE_SEMANTIQUE = 2
SOURCE_CARACTERES = 4
SOURCE_LEXICALE = 8

LIBELLES_SOURCES = {
    SOURCE_MEME_ROME: "meme_rome",
    SOURCE_SEMANTIQUE: "voisin_semantique",
    SOURCE_CARACTERES: "voisin_caracteres",
    SOURCE_LEXICALE: "voisin_lexical",
}


def ajouter_candidat(
    dictionnaire: dict[tuple[int, int], int],
    index_a: int,
    index_b: int,
    source: int,
) -> None:
    """Ajoute une paire non orientée ou complète ses sources de candidature."""

    if index_a == index_b:
        return
    paire = (min(int(index_a), int(index_b)), max(int(index_a), int(index_b)))
    dictionnaire[paire] = dictionnaire.get(paire, 0) | source


def ajouter_voisins(
    candidats: dict[tuple[int, int], int],
    matrice: object,
    nb_voisins: int,
    source: int,
    indices_globaux: np.ndarray | None = None,
) -> None:
    """Ajoute les k voisins cosinus de chaque ligne de la matrice."""

    if indices_globaux is None:
        indices_globaux = np.arange(matrice.shape[0], dtype=np.int32)

    if len(indices_globaux) <= 1:
        return

    sous_matrice = matrice[indices_globaux]
    k_reel = min(nb_voisins + 1, len(indices_globaux))

    moteur = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    moteur.fit(sous_matrice)
    voisins = moteur.kneighbors(
        sous_matrice,
        n_neighbors=k_reel,
        return_distance=False,
    )

    for position_locale, voisins_locaux in enumerate(voisins):
        index_global = int(indices_globaux[position_locale])
        nombre_ajoute = 0

        for voisin_local in voisins_locaux:
            voisin_global = int(indices_globaux[int(voisin_local)])
            if voisin_global == index_global:
                continue

            ajouter_candidat(
                candidats,
                index_global,
                voisin_global,
                source,
            )
            nombre_ajoute += 1
            if nombre_ajoute >= nb_voisins:
                break


def decoder_sources(masque: int) -> str:
    """Transforme le masque technique en liste lisible et stable."""

    return "|".join(
        libelle
        for bit, libelle in LIBELLES_SOURCES.items()
        if int(masque) & bit
    )


def generer_candidats(
    table: pd.DataFrame,
    matrice_caracteres: sparse.csr_matrix,
    matrice_lexicale: sparse.csr_matrix,
    matrice_semantique: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Génère et déduplique toutes les paires issues des quatre méthodes."""

    candidats: dict[tuple[int, int], int] = {}

    # Porte 1 : toutes les paires appartenant à un même code ROME.
    for indices in table.groupby("code_rome", sort=True).indices.values():
        indices_tries = sorted(int(index) for index in indices)
        for index_a, index_b in combinations(indices_tries, 2):
            ajouter_candidat(
                candidats,
                index_a,
                index_b,
                SOURCE_MEME_ROME,
            )
    print(f"Après même ROME       : {len(candidats):,} paires")

    # Porte 2 : voisins par le sens. Les vecteurs nuls sont exclus afin de ne
    # pas générer des voisins arbitraires.
    normes_semantiques = np.linalg.norm(matrice_semantique, axis=1)
    indices_semantiques = np.flatnonzero(normes_semantiques > 0).astype(np.int32)
    ajouter_voisins(
        candidats,
        matrice_semantique,
        NB_VOISINS_SEMANTIQUES,
        SOURCE_SEMANTIQUE,
        indices_semantiques,
    )
    print(f"Après sens            : {len(candidats):,} paires")

    # Porte 3 : voisins par groupes de caractères.
    ajouter_voisins(
        candidats,
        matrice_caracteres,
        NB_VOISINS_CARACTERES,
        SOURCE_CARACTERES,
    )
    print(f"Après caractères      : {len(candidats):,} paires")

    # Porte 4 : voisins par mots et bigrammes lemmatisés.
    ajouter_voisins(
        candidats,
        matrice_lexicale,
        NB_VOISINS_LEXICAUX,
        SOURCE_LEXICALE,
    )
    print(f"Après lexique          : {len(candidats):,} paires")

    paires_triees = sorted(candidats)
    tableau_paires = np.asarray(paires_triees, dtype=np.int32)
    masques = np.asarray(
        [candidats[paire] for paire in paires_triees],
        dtype=np.uint8,
    )

    return tableau_paires, masques


# %%
# =============================================================================
# ÉTAPE 13 — CALCUL DE WOLF, DES CONFLITS ET DES SCORES
# =============================================================================


def calculer_paires_scorees(
    table: pd.DataFrame,
    matrice_caracteres: sparse.csr_matrix,
    matrice_lexicale: sparse.csr_matrix,
    matrice_semantique: np.ndarray,
    tableau_paires: np.ndarray,
    masques_sources: np.ndarray,
    wolf: object | None,
    wolf_actif: bool,
) -> pd.DataFrame:
    """Calcule tous les indices auditables de chaque paire candidate."""

    @lru_cache(maxsize=None)
    def synsets_wolf(terme: str) -> frozenset[str]:
        if not wolf_actif or wolf is None or not terme:
            return frozenset()

        variantes = {
            terme.strip().lower(),
            terme.strip().lower().replace("-", " "),
            sans_accents(terme.strip().lower()),
        }
        identifiants: set[str] = set()

        for variante in variantes:
            if not variante:
                continue
            try:
                for synset in wolf.synsets(variante):
                    identifiants.add(str(synset.id))
            except Exception:
                # Une forme absente du lexique n'est pas une erreur métier.
                continue

        return frozenset(identifiants)

    morceaux: list[pd.DataFrame] = []
    nombre_paires = len(tableau_paires)

    for debut in range(0, nombre_paires, TAILLE_BATCH_SCORES):
        fin = min(debut + TAILLE_BATCH_SCORES, nombre_paires)
        bloc_paires = tableau_paires[debut:fin]
        index_a = bloc_paires[:, 0]
        index_b = bloc_paires[:, 1]

        # Les matrices TF-IDF sont normalisées : le produit scalaire est donc
        # directement la similarité cosinus.
        score_caracteres = np.asarray(
            matrice_caracteres[index_a]
            .multiply(matrice_caracteres[index_b])
            .sum(axis=1)
        ).ravel()

        score_lexical = np.asarray(
            matrice_lexicale[index_a]
            .multiply(matrice_lexicale[index_b])
            .sum(axis=1)
        ).ravel()

        score_semantique = np.einsum(
            "ij,ij->i",
            matrice_semantique[index_a],
            matrice_semantique[index_b],
        )

        # Une similarité cosinus sémantique peut être négative. Pour le score
        # composite, les quatre composantes sont ramenées entre 0 et 1.
        score_semantique = np.clip(score_semantique, 0.0, 1.0)
        score_caracteres = np.clip(score_caracteres, 0.0, 1.0)
        score_lexical = np.clip(score_lexical, 0.0, 1.0)

        lignes: list[dict[str, object]] = []

        for position_dans_bloc, (a, b) in enumerate(bloc_paires):
            a = int(a)
            b = int(b)
            ligne_a = table.iloc[a]
            ligne_b = table.iloc[b]

            termes_a = set(str(ligne_a["texte_lemmatise"]).split())
            termes_b = set(str(ligne_b["texte_lemmatise"]).split())
            union = termes_a | termes_b
            intersection = termes_a & termes_b
            score_jaccard = len(intersection) / len(union) if union else 0.0

            terme_a = str(ligne_a["terme_metier_initial"])
            terme_b = str(ligne_b["terme_metier_initial"])
            meme_terme = bool(terme_a and terme_a == terme_b)

            terme_wolf_a = str(ligne_a["terme_metier_initial_wolf"])
            terme_wolf_b = str(ligne_b["terme_metier_initial_wolf"])

            synonymie_wolf = False
            if terme_a and terme_b and terme_a != terme_b and wolf_actif:
                synonymie_wolf = bool(
                    synsets_wolf(terme_wolf_a)
                    & synsets_wolf(terme_wolf_b)
                )

            meme_rome = str(ligne_a["code_rome"]) == str(ligne_b["code_rome"])
            identite_normalisee = (
                str(ligne_a["libelle_normalise"])
                == str(ligne_b["libelle_normalise"])
            )

            marqueurs_a = lire_ensemble(ligne_a["marqueurs_fonction"])
            marqueurs_b = lire_ensemble(ligne_b["marqueurs_fonction"])
            divergence_marqueurs = marqueurs_a.symmetric_difference(marqueurs_b)
            conflit_qualificatif = bool(divergence_marqueurs)

            nombres_a = lire_ensemble(ligne_a["nombres"])
            nombres_b = lire_ensemble(ligne_b["nombres"])
            conflit_nombres = bool(
                (nombres_a or nombres_b) and nombres_a != nombres_b
            )

            avec_sans_a = lire_ensemble(ligne_a["marqueurs_avec_sans"])
            avec_sans_b = lire_ensemble(ligne_b["marqueurs_avec_sans"])
            conflit_avec_sans = bool(
                ("avec" in avec_sans_a and "sans" in avec_sans_b)
                or ("sans" in avec_sans_a and "avec" in avec_sans_b)
            )

            conflits: list[str] = []
            if conflit_qualificatif:
                conflits.append(
                    "qualificatifs:"
                    + ",".join(sorted(divergence_marqueurs))
                )
            if conflit_nombres:
                conflits.append(
                    "nombres:"
                    + ",".join(sorted(nombres_a))
                    + "/"
                    + ",".join(sorted(nombres_b))
                )
            if conflit_avec_sans:
                conflits.append("opposition_avec_sans")

            bonus_rome = BONUS_MEME_ROME if meme_rome else 0.0
            bonus_terme = BONUS_MEME_TERME_INITIAL if meme_terme else 0.0
            bonus_wolf = BONUS_SYNONYMIE_WOLF if synonymie_wolf else 0.0

            composite_avant_bornage = (
                POIDS_SEMANTIQUE * float(score_semantique[position_dans_bloc])
                + POIDS_CARACTERES * float(score_caracteres[position_dans_bloc])
                + POIDS_LEXICAL * float(score_lexical[position_dans_bloc])
                + POIDS_JACCARD * float(score_jaccard)
                + bonus_rome
                + bonus_terme
                + bonus_wolf
            )
            score_composite = min(1.0, max(0.0, composite_avant_bornage))

            lignes.append(
                {
                    "index_a": a,
                    "index_b": b,
                    "code_ogr_a": str(ligne_a["code_ogr"]),
                    "code_ogr_b": str(ligne_b["code_ogr"]),
                    "score_semantique": float(score_semantique[position_dans_bloc]),
                    "score_caracteres": float(score_caracteres[position_dans_bloc]),
                    "score_lexical": float(score_lexical[position_dans_bloc]),
                    "score_jaccard": float(score_jaccard),
                    "bonus_meme_rome": float(bonus_rome),
                    "bonus_meme_terme_initial": float(bonus_terme),
                    "bonus_synonymie_wolf": float(bonus_wolf),
                    "score_composite_avant_bornage": float(composite_avant_bornage),
                    "score_composite": float(score_composite),
                    "identite_normalisee": identite_normalisee,
                    "meme_code_rome": meme_rome,
                    "inter_rome": not meme_rome,
                    "meme_terme_metier_initial": meme_terme,
                    "synonymie_wolf": synonymie_wolf,
                    "wolf_actif": wolf_actif,
                    "conflit_qualificatif": conflit_qualificatif,
                    "conflit_nombres": conflit_nombres,
                    "conflit_avec_sans": conflit_avec_sans,
                    "conflits_detectes": "|".join(conflits),
                    "source_candidature": decoder_sources(
                        int(masques_sources[debut + position_dans_bloc])
                    ),
                }
            )

        morceaux.append(pd.DataFrame(lignes))
        print(f"Paires évaluées : {fin:,}/{nombre_paires:,}")

    resultat = pd.concat(morceaux, ignore_index=True)

    colonnes_numeriques = [
        "score_semantique",
        "score_caracteres",
        "score_lexical",
        "score_jaccard",
        "bonus_meme_rome",
        "bonus_meme_terme_initial",
        "bonus_synonymie_wolf",
        "score_composite_avant_bornage",
        "score_composite",
    ]
    resultat[colonnes_numeriques] = resultat[colonnes_numeriques].round(6)

    return resultat


# %%
# =============================================================================
# ÉTAPE 14 — CRÉATION OU RECHARGEMENT DU CACHE DES SCORES
# =============================================================================

afficher_titre(14, "GÉNÉRATION DES CANDIDATS ET CALCUL DES SCORES")

if CHEMIN_PAIRES_SCOREES.exists() and not FORCER_RECALCUL_COMPLET:
    print("Réutilisation du cache des scores :", CHEMIN_PAIRES_SCOREES)
    paires_scorees = pd.read_parquet(CHEMIN_PAIRES_SCOREES)
    wolf_actif = bool(paires_scorees["wolf_actif"].any())
else:
    debut_scores = time.perf_counter()
    wolf, wolf_actif = initialiser_wolf()

    matrice_caracteres, matrice_lexicale = construire_matrices_tfidf(
        appellations
    )

    tableau_paires, masques_sources = generer_candidats(
        appellations,
        matrice_caracteres,
        matrice_lexicale,
        vecteurs_semantiques,
    )

    print(f"Nombre final de paires candidates : {len(tableau_paires):,}")

    paires_scorees = calculer_paires_scorees(
        appellations,
        matrice_caracteres,
        matrice_lexicale,
        vecteurs_semantiques,
        tableau_paires,
        masques_sources,
        wolf,
        wolf_actif,
    )

    paires_scorees.to_parquet(
        CHEMIN_PAIRES_SCOREES,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    afficher_duree(debut_scores, "Calcul des scores terminé")
    print("Cache écrit :", CHEMIN_PAIRES_SCOREES)

print("Paires disponibles pour classement :", f"{len(paires_scorees):,}")
print("Bonus WOLF actif                    :", wolf_actif)


# %%
# =============================================================================
# ÉTAPE 15 — CLASSEMENT DES PAIRES DANS LES TROIS NIVEAUX
# =============================================================================
#
# Les conditions sont volontairement explicites. Une paire est testée d'abord
# au niveau FORTE, puis PLAUSIBLE, puis A_ARBITRER. Elle ne peut donc apparaître
# que dans un seul niveau.
# =============================================================================


def classer_paires(table: pd.DataFrame) -> pd.DataFrame:
    """Attribue un niveau, une règle de classement et un score de confiance."""

    resultat = table.copy()

    aucun_conflit = ~(
        resultat["conflit_qualificatif"]
        | resultat["conflit_nombres"]
        | resultat["conflit_avec_sans"]
    )

    # FORTE — identité normalisée, même entre deux ROME.
    forte_identite = resultat["identite_normalisee"] & aucun_conflit

    # FORTE — même ROME et convergence élevée des trois grandes mesures.
    forte_convergence = (
        resultat["meme_code_rome"]
        & aucun_conflit
        & (resultat["score_composite"] >= SEUIL_FORTE_MEME_ROME)
        & (resultat["score_semantique"] >= 0.72)
        & (resultat["score_caracteres"] >= 0.76)
        & (resultat["score_lexical"] >= 0.66)
    )
    est_forte = forte_identite | forte_convergence

    # PLAUSIBLE — WOLF, terme initial commun ou convergence générale dans un
    # même ROME, toujours sans conflit métier identifié.
    plausible_meme_rome = (
        resultat["meme_code_rome"]
        & aucun_conflit
        & (
            (
                resultat["synonymie_wolf"]
                & (resultat["score_composite"] >= 0.66)
            )
            | (
                resultat["meme_terme_metier_initial"]
                & (resultat["score_caracteres"] >= 0.68)
                & (resultat["score_lexical"] >= 0.54)
                & (resultat["score_composite"] >= SEUIL_PLAUSIBLE_MEME_ROME)
            )
            | (
                (resultat["score_composite"] >= 0.74)
                & (resultat["score_semantique"] >= 0.68)
                & (resultat["score_caracteres"] >= 0.60)
                & (resultat["score_lexical"] >= 0.50)
            )
        )
    )

    # Les rapprochements inter-ROME exigent davantage de convergence et au
    # moins un appui lexical explicite.
    plausible_inter_rome = (
        resultat["inter_rome"]
        & aucun_conflit
        & (resultat["score_composite"] >= SEUIL_PLAUSIBLE_INTER_ROME)
        & (resultat["score_semantique"] >= 0.74)
        & (resultat["score_caracteres"] >= 0.64)
        & (resultat["score_lexical"] >= 0.56)
        & (
            resultat["meme_terme_metier_initial"]
            | resultat["synonymie_wolf"]
            | (resultat["score_jaccard"] >= 0.45)
        )
    )
    est_plausible = (~est_forte) & (
        plausible_meme_rome | plausible_inter_rome
    )

    # Nombre d'indices significatifs. Ce compteur évite de retenir au niveau
    # « à arbitrer » une paire portée par une seule ressemblance fragile.
    nb_indices_forts = (
        (resultat["score_semantique"] >= 0.66).astype(int)
        + (resultat["score_caracteres"] >= 0.55).astype(int)
        + (resultat["score_lexical"] >= 0.45).astype(int)
        + (resultat["score_jaccard"] >= 0.30).astype(int)
        + resultat["meme_terme_metier_initial"].astype(int)
        + resultat["synonymie_wolf"].astype(int)
    )

    arbitrer_standard = (
        (resultat["score_composite"] >= SEUIL_ARBITRER_STANDARD)
        & (nb_indices_forts >= 2)
    )
    arbitrer_semantique = (
        (resultat["score_semantique"] >= 0.82)
        & (
            resultat[["score_caracteres", "score_lexical"]].max(axis=1)
            >= 0.30
        )
    )
    arbitrer_inter_rome = (
        resultat["inter_rome"]
        & (resultat["score_composite"] >= SEUIL_ARBITRER_INTER_ROME)
        & (nb_indices_forts >= 3)
    )
    est_arbitrer = (~est_forte) & (~est_plausible) & (
        arbitrer_standard
        | arbitrer_semantique
        | arbitrer_inter_rome
    )

    resultat["niveau_preconisation"] = "NON_RETENUE"
    resultat.loc[est_arbitrer, "niveau_preconisation"] = "A_ARBITRER"
    resultat.loc[est_plausible, "niveau_preconisation"] = "PLAUSIBLE"
    resultat.loc[est_forte, "niveau_preconisation"] = "FORTE"

    resultat["regle_classement"] = "score_insuffisant"
    resultat.loc[est_arbitrer, "regle_classement"] = (
        "plusieurs_indices_a_instruire"
    )
    resultat.loc[
        est_arbitrer & arbitrer_semantique,
        "regle_classement",
    ] = "forte_proximite_semantique_a_instruire"
    resultat.loc[
        est_arbitrer & arbitrer_inter_rome,
        "regle_classement",
    ] = "rapprochement_inter_rome_a_instruire"
    resultat.loc[
        est_plausible & plausible_meme_rome,
        "regle_classement",
    ] = "convergence_dans_un_meme_rome"
    resultat.loc[
        est_plausible & plausible_inter_rome,
        "regle_classement",
    ] = "forte_convergence_inter_rome"
    resultat.loc[
        est_forte & forte_convergence,
        "regle_classement",
    ] = "forte_convergence_dans_un_meme_rome"
    resultat.loc[
        est_forte & forte_identite,
        "regle_classement",
    ] = "identite_apres_normalisation"

    # Le score de confiance classe les propositions à l'intérieur d'un niveau.
    # Il ne s'agit pas d'une probabilité.
    confiance = np.zeros(len(resultat), dtype=np.int16)
    composite = resultat["score_composite"].to_numpy(dtype=float)

    masque = resultat["niveau_preconisation"].eq("FORTE").to_numpy()
    confiance[masque] = np.rint(
        88 + np.clip((composite[masque] - 0.72) / 0.28, 0, 1) * 11
    ).astype(np.int16)

    masque = resultat["niveau_preconisation"].eq("PLAUSIBLE").to_numpy()
    confiance[masque] = np.rint(
        65 + np.clip((composite[masque] - 0.62) / 0.28, 0, 1) * 22
    ).astype(np.int16)

    masque = resultat["niveau_preconisation"].eq("A_ARBITRER").to_numpy()
    confiance[masque] = np.rint(
        45 + np.clip((composite[masque] - 0.45) / 0.35, 0, 1) * 27
    ).astype(np.int16)

    resultat["score_confiance"] = confiance
    resultat["nb_indices_forts"] = nb_indices_forts.astype(np.int8)

    return resultat


paires_classees = classer_paires(paires_scorees)

print("Répartition avant retrait des non-retenues :")
print(paires_classees["niveau_preconisation"].value_counts().to_string())

paires_retenues = (
    paires_classees
    .loc[paires_classees["niveau_preconisation"] != "NON_RETENUE"]
    .copy()
    .reset_index(drop=True)
)

if paires_retenues.empty:
    raise RuntimeError(
        "Aucune paire n'a franchi les seuils. Vérifier les scores et paramètres."
    )


# %%
# =============================================================================
# ÉTAPE 16 — CHOIX DU PARENT POUR CHAQUE PAIRE
# =============================================================================
#
# Le choix est déterministe. Il privilégie successivement :
#   1. une appellation non marquée peu usitée ;
#   2. une appellation principale ;
#   3. un libellé plus générique et plus court ;
#   4. l'absence de marqueur subordonné ;
#   5. le code OGR comme départage stable.
# =============================================================================


def cle_choix_parent(ligne: pd.Series) -> tuple[object, ...]:
    """Construit la clé lexicographique utilisée pour choisir le parent."""

    termes = str(ligne["texte_lemmatise"]).split()
    marqueurs = lire_ensemble(ligne["marqueurs_fonction"])
    contient_subordonne = bool(marqueurs & MARQUEURS_SUBORDONNES)

    return (
        bool(ligne["peu_usite_bool"]),
        str(ligne["classification"]) != "PRINCIPALE",
        max(0, len(termes) - 1),
        len(str(ligne["libelle_normalise"])),
        contient_subordonne,
        str(ligne["code_ogr"]),
    )


def expliquer_choix_parent(
    parent: pd.Series,
    membre: pd.Series,
) -> str:
    """Indique le premier critère ayant départagé les deux appellations."""

    if bool(parent["peu_usite_bool"]) != bool(membre["peu_usite_bool"]):
        return "parent_non_peu_usite"
    if str(parent["classification"]) != str(membre["classification"]):
        return "appellation_principale_privilegiee"

    nb_parent = len(str(parent["texte_lemmatise"]).split())
    nb_membre = len(str(membre["texte_lemmatise"]).split())
    if nb_parent != nb_membre:
        return "libelle_plus_generique"

    sous_parent = bool(
        lire_ensemble(parent["marqueurs_fonction"])
        & MARQUEURS_SUBORDONNES
    )
    sous_membre = bool(
        lire_ensemble(membre["marqueurs_fonction"])
        & MARQUEURS_SUBORDONNES
    )
    if sous_parent != sous_membre:
        return "absence_de_marqueur_subordonne"

    return "departage_stable_par_code_ogr"


parents: list[int] = []
membres: list[int] = []
raisons_parent: list[str] = []

for paire in paires_retenues.itertuples(index=False):
    index_a = int(paire.index_a)
    index_b = int(paire.index_b)
    ligne_a = appellations.iloc[index_a]
    ligne_b = appellations.iloc[index_b]

    if cle_choix_parent(ligne_a) <= cle_choix_parent(ligne_b):
        index_parent, index_membre = index_a, index_b
        ligne_parent, ligne_membre = ligne_a, ligne_b
    else:
        index_parent, index_membre = index_b, index_a
        ligne_parent, ligne_membre = ligne_b, ligne_a

    parents.append(index_parent)
    membres.append(index_membre)
    raisons_parent.append(expliquer_choix_parent(ligne_parent, ligne_membre))

paires_retenues["index_parent"] = np.asarray(parents, dtype=np.int32)
paires_retenues["index_membre"] = np.asarray(membres, dtype=np.int32)
paires_retenues["regle_choix_parent"] = raisons_parent


# %%
# =============================================================================
# ÉTAPE 17 — CONSTRUCTION DES LIGNES DE FAMILLES
# =============================================================================
#
# Une famille suggérée est définie par un même parent et un même niveau. Toutes
# les paires orientées vers ce parent sont regroupées sous le même
# id_famille_suggeree. Une ligne représente un rattachement parent → membre.
# =============================================================================

date_generation_utc = datetime.now(timezone.utc).replace(
    microsecond=0
).isoformat()

lignes_sortie: list[dict[str, object]] = []

for paire in paires_retenues.itertuples(index=False):
    parent = appellations.iloc[int(paire.index_parent)]
    membre = appellations.iloc[int(paire.index_membre)]
    niveau = str(paire.niveau_preconisation)

    id_famille = identifiant_stable(
        VERSION_METHODE,
        niveau,
        parent["code_ogr"],
        longueur=18,
    )
    id_suggestion = identifiant_stable(
        VERSION_METHODE,
        niveau,
        parent["code_ogr"],
        membre["code_ogr"],
        longueur=22,
    )

    lignes_sortie.append(
        {
            "id_suggestion": id_suggestion,
            "id_famille_suggeree": id_famille,
            "niveau_preconisation": niveau,
            "version_preconisation": VERSION_METHODE,
            "version_rome": VERSION_ROME,
            "date_generation_utc": date_generation_utc,
            "code_ogr_parent": str(parent["code_ogr"]),
            "libelle_parent": str(parent["libelle_appellation_long"]),
            "code_rome_parent": str(parent["code_rome"]),
            "classification_parent": str(parent["classification"]),
            "peu_usite_parent": bool(parent["peu_usite_bool"]),
            "code_ogr_membre": str(membre["code_ogr"]),
            "libelle_membre": str(membre["libelle_appellation_long"]),
            "code_rome_membre": str(membre["code_rome"]),
            "classification_membre": str(membre["classification"]),
            "peu_usite_membre": bool(membre["peu_usite_bool"]),
            "inter_rome": bool(paire.inter_rome),
            "score_confiance": int(paire.score_confiance),
            "score_composite": float(paire.score_composite),
            "score_composite_avant_bornage": float(
                paire.score_composite_avant_bornage
            ),
            "score_semantique": float(paire.score_semantique),
            "score_caracteres": float(paire.score_caracteres),
            "score_lexical": float(paire.score_lexical),
            "score_jaccard": float(paire.score_jaccard),
            "bonus_meme_rome": float(paire.bonus_meme_rome),
            "bonus_meme_terme_initial": float(
                paire.bonus_meme_terme_initial
            ),
            "bonus_synonymie_wolf": float(paire.bonus_synonymie_wolf),
            "identite_normalisee": bool(paire.identite_normalisee),
            "meme_code_rome": bool(paire.meme_code_rome),
            "meme_terme_metier_initial": bool(
                paire.meme_terme_metier_initial
            ),
            "synonymie_wolf": bool(paire.synonymie_wolf),
            "wolf_actif": bool(paire.wolf_actif),
            "conflit_qualificatif": bool(paire.conflit_qualificatif),
            "conflit_nombres": bool(paire.conflit_nombres),
            "conflit_avec_sans": bool(paire.conflit_avec_sans),
            "conflits_detectes": str(paire.conflits_detectes),
            "nb_indices_forts": int(paire.nb_indices_forts),
            "source_candidature": str(paire.source_candidature),
            "regle_classement": str(paire.regle_classement),
            "regle_choix_parent": str(paire.regle_choix_parent),
        }
    )

preconisations = pd.DataFrame(lignes_sortie)

# Tri reproductible et calcul du rang des membres dans chaque famille.
preconisations = preconisations.sort_values(
    [
        "niveau_preconisation",
        "id_famille_suggeree",
        "score_confiance",
        "score_composite",
        "code_ogr_membre",
    ],
    ascending=[True, True, False, False, True],
    kind="stable",
).reset_index(drop=True)

preconisations["rang_dans_famille"] = (
    preconisations.groupby("id_famille_suggeree").cumcount() + 1
).astype(np.int32)

preconisations["nb_membres_famille"] = (
    preconisations.groupby("id_famille_suggeree")["id_suggestion"]
    .transform("size")
    .astype(np.int32)
)

# Ordre final : informations directement utiles dans OGRE, puis détail du score.
ordre_colonnes = [
    "id_suggestion",
    "id_famille_suggeree",
    "niveau_preconisation",
    "version_preconisation",
    "version_rome",
    "date_generation_utc",
    "rang_dans_famille",
    "nb_membres_famille",
    "code_ogr_parent",
    "libelle_parent",
    "code_rome_parent",
    "classification_parent",
    "peu_usite_parent",
    "code_ogr_membre",
    "libelle_membre",
    "code_rome_membre",
    "classification_membre",
    "peu_usite_membre",
    "inter_rome",
    "score_confiance",
    "score_composite",
    "score_composite_avant_bornage",
    "score_semantique",
    "score_caracteres",
    "score_lexical",
    "score_jaccard",
    "bonus_meme_rome",
    "bonus_meme_terme_initial",
    "bonus_synonymie_wolf",
    "identite_normalisee",
    "meme_code_rome",
    "meme_terme_metier_initial",
    "synonymie_wolf",
    "wolf_actif",
    "conflit_qualificatif",
    "conflit_nombres",
    "conflit_avec_sans",
    "conflits_detectes",
    "nb_indices_forts",
    "source_candidature",
    "regle_classement",
    "regle_choix_parent",
]
preconisations = preconisations[ordre_colonnes]


# %%
# =============================================================================
# ÉTAPE 18 — CONTRÔLES DE COHÉRENCE DES RÉSULTATS
# =============================================================================

afficher_titre(18, "CONTRÔLES DES PRÉCONISATIONS")

assert not preconisations.empty
assert not preconisations["id_suggestion"].duplicated().any()
assert (
    preconisations["code_ogr_parent"]
    != preconisations["code_ogr_membre"]
).all()

niveaux_valides = {"FORTE", "PLAUSIBLE", "A_ARBITRER"}
assert set(preconisations["niveau_preconisation"]) <= niveaux_valides

bornes_confiance = {
    "FORTE": (88, 99),
    "PLAUSIBLE": (65, 87),
    "A_ARBITRER": (45, 72),
}

for niveau, (minimum, maximum) in bornes_confiance.items():
    valeurs = preconisations.loc[
        preconisations["niveau_preconisation"] == niveau,
        "score_confiance",
    ]
    if not valeurs.empty:
        assert valeurs.between(minimum, maximum).all()

# Vérification de l'exclusivité des paires entre les trois niveaux.
cles_paires = preconisations.apply(
    lambda ligne: "|".join(
        sorted([ligne["code_ogr_parent"], ligne["code_ogr_membre"]])
    ),
    axis=1,
)
assert not cles_paires.duplicated().any(), (
    "Une même paire apparaît dans plusieurs propositions ou niveaux."
)

# Un identifiant de famille correspond toujours à un seul parent et niveau.
controle_familles = preconisations.groupby("id_famille_suggeree").agg(
    nb_parent=("code_ogr_parent", "nunique"),
    nb_niveau=("niveau_preconisation", "nunique"),
)
assert (controle_familles["nb_parent"] == 1).all()
assert (controle_familles["nb_niveau"] == 1).all()

print("Contrôles d'unicité, de bornes et de familles : OK")


# %%
# =============================================================================
# ÉTAPE 19 — CRÉATION DES TROIS TABLES ET EXPORT PARQUET
# =============================================================================

afficher_titre(19, "EXPORT DES TROIS TABLES PARQUET")

tables_sortie = {
    "FORTE": (
        CHEMIN_SORTIE_FORTE,
        preconisations.loc[
            preconisations["niveau_preconisation"] == "FORTE"
        ].copy(),
    ),
    "PLAUSIBLE": (
        CHEMIN_SORTIE_PLAUSIBLE,
        preconisations.loc[
            preconisations["niveau_preconisation"] == "PLAUSIBLE"
        ].copy(),
    ),
    "A_ARBITRER": (
        CHEMIN_SORTIE_ARBITRER,
        preconisations.loc[
            preconisations["niveau_preconisation"] == "A_ARBITRER"
        ].copy(),
    ),
}

resume_tables: dict[str, dict[str, int]] = {}

for niveau, (chemin, table) in tables_sortie.items():
    table.to_parquet(
        chemin,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    # Relecture immédiate : elle vérifie que le Parquet est bien exploitable.
    table_relue = pd.read_parquet(chemin, engine="pyarrow")
    assert len(table_relue) == len(table)
    assert table_relue.columns.tolist() == table.columns.tolist()

    resume_tables[niveau] = {
        "lignes": int(len(table)),
        "familles": int(table["id_famille_suggeree"].nunique()),
        "parents_distincts": int(table["code_ogr_parent"].nunique()),
        "membres_distincts": int(table["code_ogr_membre"].nunique()),
        "lignes_inter_rome": int(table["inter_rome"].sum()),
    }

    print(
        f"{niveau:<12} : {len(table):>7,} lignes | "
        f"{table['id_famille_suggeree'].nunique():>6,} familles | {chemin}"
    )


# %%
# =============================================================================
# ÉTAPE 20 — MANIFESTE DE REPRODUCTIBILITÉ
# =============================================================================
#
# Le manifeste conserve les paramètres et versions sans modifier le schéma des
# trois tables. Il permet de comprendre ultérieurement comment les fichiers ont
# été produits.
# =============================================================================

manifeste = {
    "version_methode": VERSION_METHODE,
    "version_rome": VERSION_ROME,
    "date_generation_utc": date_generation_utc,
    "source": {
        "url": URL_FRANCE_TRAVAIL,
        "fichier": NOM_FICHIER_SOURCE,
        "nombre_ogr": nb_ogr,
        "nombre_rome": nb_rome,
    },
    "environnement": {
        "python": sys.version.splitlines()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit_learn": sklearn.__version__,
        "spacy": spacy.__version__,
        "modele_spacy": MODELE_SPACY,
        "wolf_spec": WOLF_SPEC,
        "wolf_actif": wolf_actif,
    },
    "voisins": {
        "semantiques": NB_VOISINS_SEMANTIQUES,
        "caracteres": NB_VOISINS_CARACTERES,
        "lexicaux": NB_VOISINS_LEXICAUX,
    },
    "poids": {
        "semantique": POIDS_SEMANTIQUE,
        "caracteres": POIDS_CARACTERES,
        "lexical": POIDS_LEXICAL,
        "jaccard": POIDS_JACCARD,
    },
    "bonus": {
        "meme_rome": BONUS_MEME_ROME,
        "meme_terme_initial": BONUS_MEME_TERME_INITIAL,
        "synonymie_wolf": BONUS_SYNONYMIE_WOLF,
    },
    "seuils": {
        "forte_meme_rome": SEUIL_FORTE_MEME_ROME,
        "plausible_meme_rome": SEUIL_PLAUSIBLE_MEME_ROME,
        "plausible_inter_rome": SEUIL_PLAUSIBLE_INTER_ROME,
        "arbitrer_standard": SEUIL_ARBITRER_STANDARD,
        "arbitrer_inter_rome": SEUIL_ARBITRER_INTER_ROME,
    },
    "resultats": resume_tables,
    "avertissement": (
        "Les scores classent des pistes. Ils ne sont pas des probabilités "
        "et ne remplacent pas la décision métier."
    ),
}

CHEMIN_MANIFESTE.write_text(
    json.dumps(manifeste, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("Manifeste écrit :", CHEMIN_MANIFESTE)


# %%
# =============================================================================
# ÉTAPE 21 — ENVOI FACULTATIF VERS S3 SSPCLOUD
# =============================================================================
#
# s3fs reprend automatiquement les identifiants AWS/MinIO disponibles dans le
# service Onyxia. Le point d'accès est recherché dans plusieurs variables
# usuelles pour rester compatible avec les images de service SSPCloud.
# =============================================================================


def normaliser_endpoint_s3(endpoint: str | None) -> str | None:
    """Ajoute https:// lorsqu'un endpoint ne contient pas de protocole."""

    if not endpoint:
        return None
    endpoint = endpoint.strip()
    if not endpoint.startswith(("http://", "https://")):
        endpoint = "https://" + endpoint
    return endpoint.rstrip("/")


def creer_systeme_s3() -> s3fs.S3FileSystem:
    """Crée un accès S3 à partir de l'environnement courant."""

    endpoint = normaliser_endpoint_s3(
        os.getenv("AWS_S3_ENDPOINT")
        or os.getenv("AWS_ENDPOINT_URL_S3")
        or os.getenv("AWS_ENDPOINT_URL")
        or os.getenv("S3_ENDPOINT_URL")
    )

    options: dict[str, object] = {}
    if endpoint:
        options["client_kwargs"] = {"endpoint_url": endpoint}
        print("Endpoint S3 utilisé :", endpoint)
    else:
        print("Endpoint S3 : configuration AWS/Onyxia par défaut")

    return s3fs.S3FileSystem(**options)


if ENVOYER_VERS_S3:
    afficher_titre(21, "ENVOI DES LIVRABLES VERS S3")
    systeme_s3 = creer_systeme_s3()
    prefixe_s3 = DOSSIER_S3.removeprefix("s3://").rstrip("/")

    fichiers_a_envoyer = [
        CHEMIN_SORTIE_FORTE,
        CHEMIN_SORTIE_PLAUSIBLE,
        CHEMIN_SORTIE_ARBITRER,
        CHEMIN_MANIFESTE,
    ]

    for chemin_local in fichiers_a_envoyer:
        cible = f"{prefixe_s3}/{chemin_local.name}"
        systeme_s3.put(str(chemin_local), cible)

        if not systeme_s3.exists(cible):
            raise RuntimeError(f"Fichier S3 introuvable après envoi : {cible}")

        print("Envoyé :", f"s3://{cible}")
else:
    print("\nEnvoi S3 non demandé.")
    print(
        "Après contrôle, relancer avec OGRE_UPLOAD_S3=1 pour envoyer vers :\n"
        f"{DOSSIER_S3}/"
    )


# %%
# =============================================================================
# ÉTAPE 22 — SYNTHÈSE FINALE
# =============================================================================

afficher_titre(22, "TRAITEMENT TERMINÉ")

for niveau in ("FORTE", "PLAUSIBLE", "A_ARBITRER"):
    statistiques = resume_tables[niveau]
    print(
        f"{niveau:<12} | "
        f"lignes={statistiques['lignes']:,} | "
        f"familles={statistiques['familles']:,} | "
        f"membres={statistiques['membres_distincts']:,} | "
        f"inter_ROME={statistiques['lignes_inter_rome']:,}"
    )

print("\nLivrables locaux :")
print(" -", CHEMIN_SORTIE_FORTE)
print(" -", CHEMIN_SORTIE_PLAUSIBLE)
print(" -", CHEMIN_SORTIE_ARBITRER)
print(" -", CHEMIN_MANIFESTE)
print("\nProgramme OGRE terminé avec succès.")
