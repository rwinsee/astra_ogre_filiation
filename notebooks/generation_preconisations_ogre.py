# %%
# =============================================================================
# OGRE — GÉNÉRATION DES PRÉCONISATIONS DE FAMILLES OGR
# =============================================================================
#
# Objectif :
#   produire trois tables de préconisations :
#   1. familles fortes ;
#   2. familles plausibles ;
#   3. rapprochements à arbitrer.
#
# Le fichier est organisé en blocs "# %%" exécutables séparément dans VSCode.
# Il peut également être relancé intégralement avec :
#
# /opt/python/bin/python \
#   /home/onyxia/work/astra_ogre_filiation/notebooks/\
# generation_preconisations_ogre.py
#
# Les données téléchargées et les Parquet produits sont placés dans data/.
# Ce dossier ne doit pas être envoyé sur GitHub.
# =============================================================================


# %%
# =============================================================================
# ÉTAPE 0 — IMPORTS GÉNÉRAUX
# =============================================================================
#
# Tous les imports sont regroupés ici pour éviter de les répéter dans les
# différents blocs.
# =============================================================================

import io
import shutil
import sys
import zipfile
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


# %%
# =============================================================================
# ÉTAPE 1 — PARAMÈTRES DU TRAITEMENT
# =============================================================================
#
# Ces paramètres permettent de modifier facilement :
#   - la version du référentiel ;
#   - l’adresse de téléchargement ;
#   - les dossiers de travail ;
#   - les volumes attendus ;
#   - le comportement en cas de relance.
# =============================================================================

VERSION_ROME = "4.61"
VERSION_ROME_FICHIER = "v461"

URL_FRANCE_TRAVAIL = (
    "https://api.francetravail.fr/"
    "api-nomenclatureemploi/v1/open-data/csv"
)

# Le script se trouve dans notebooks/.
# parents[1] désigne donc la racine du projet astra_ogre_filiation.
RACINE_PROJET = Path(__file__).resolve().parents[1]

DOSSIER_SOURCE = RACINE_PROJET / "data" / "raw"
DOSSIER_SORTIE = RACINE_PROJET / "data" / "output"

NOM_FICHIER_SOURCE = (
    "unix_referentiel_appellation_v461_utf8.csv"
)

CHEMIN_SOURCE = DOSSIER_SOURCE / NOM_FICHIER_SOURCE

# Volumes attendus pour le référentiel ROME 4.61.
NB_OGR_ATTENDU = 14_301
NB_ROME_ATTENDU = 1_911
NB_PRINCIPALES_ATTENDU = 1_911
NB_SYNONYMES_ATTENDU = 12_390

# False : réutiliser le fichier déjà téléchargé.
# True  : retélécharger et remplacer le fichier local.
FORCER_TELECHARGEMENT = False

DOSSIER_SOURCE.mkdir(parents=True, exist_ok=True)
DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)


# %%
# =============================================================================
# ÉTAPE 2 — CONTRÔLE DE L’ENVIRONNEMENT PYTHON
# =============================================================================
#
# Ce contrôle permet de conserver une trace des versions utilisées.
# En cas d’évolution future des résultats, ces informations faciliteront
# l’identification d’un changement de bibliothèque.
# =============================================================================

print("\n" + "=" * 72)
print("ÉTAPE 2 — CONTRÔLE DE L’ENVIRONNEMENT")
print("=" * 72)

print("Python :", sys.version)
print("Dossier de travail :", Path.cwd())
print("Racine du projet :", RACINE_PROJET)

print("\npandas :", pd.__version__)
print("numpy :", np.__version__)
print("pyarrow :", pyarrow.__version__)
print("scikit-learn :", sklearn.__version__)
print("spaCy :", spacy.__version__)

print("\nEnvironnement Python OGRE opérationnel.")


# %%
# =============================================================================
# ÉTAPE 3 — RÉCUPÉRATION DU RÉFÉRENTIEL FRANCE TRAVAIL
# =============================================================================
#
# Lors de la première exécution :
#   - le référentiel est téléchargé depuis France Travail ;
#   - si la réponse est une archive ZIP, le fichier des appellations V4.61
#     est recherché puis extrait ;
#   - sinon, la réponse est enregistrée directement comme fichier CSV.
#
# Lors des exécutions suivantes :
#   - le fichier local est réutilisé ;
#   - aucun téléchargement n’est réalisé, sauf si
#     FORCER_TELECHARGEMENT = True.
# =============================================================================

print("\n" + "=" * 72)
print("ÉTAPE 3 — RÉCUPÉRATION DU RÉFÉRENTIEL")
print("=" * 72)

if CHEMIN_SOURCE.exists() and not FORCER_TELECHARGEMENT:

    # Le référentiel existe déjà : on le conserve pour garantir que les
    # différentes relances utilisent exactement la même source.
    chemin_csv = CHEMIN_SOURCE

    print("Référentiel local réutilisé :", chemin_csv)

else:

    print("Téléchargement depuis :", URL_FRANCE_TRAVAIL)

    reponse = requests.get(
        URL_FRANCE_TRAVAIL,
        timeout=300,
    )

    # Une erreur explicite est levée si France Travail renvoie un statut
    # HTTP d’erreur.
    reponse.raise_for_status()

    contenu = reponse.content

    if zipfile.is_zipfile(io.BytesIO(contenu)):

        # L’adresse peut renvoyer une archive contenant plusieurs ressources.
        with zipfile.ZipFile(io.BytesIO(contenu)) as archive:

            fichiers_appellations = [
                nom
                for nom in archive.namelist()
                if "referentiel_appellation" in nom.lower()
                and nom.lower().endswith(".csv")
            ]

            # On impose la version 4.61 afin de ne pas utiliser
            # involontairement une version plus récente.
            fichiers_v461 = [
                nom
                for nom in fichiers_appellations
                if VERSION_ROME_FICHIER in nom.lower()
            ]

            if not fichiers_v461:
                raise FileNotFoundError(
                    "Le fichier des appellations ROME 4.61 "
                    "n’a pas été trouvé dans l’archive.\n"
                    f"Fichiers candidats : {fichiers_appellations}"
                )

            nom_dans_archive = fichiers_v461[0]

            with archive.open(nom_dans_archive) as source:
                with CHEMIN_SOURCE.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

    else:

        # L’adresse a renvoyé directement le contenu du CSV.
        CHEMIN_SOURCE.write_bytes(contenu)

    chemin_csv = CHEMIN_SOURCE

    print("Référentiel téléchargé :", chemin_csv)

if not chemin_csv.exists():
    raise FileNotFoundError(
        f"Le référentiel est introuvable : {chemin_csv}"
    )

if chemin_csv.stat().st_size == 0:
    raise ValueError(
        f"Le référentiel est vide : {chemin_csv}"
    )

print(
    "Taille du fichier :",
    round(chemin_csv.stat().st_size / 1_000_000, 2),
    "Mo",
)


# %%
# =============================================================================
# ÉTAPE 4 — LECTURE DU RÉFÉRENTIEL
# =============================================================================
#
# sep=None et engine="python" demandent à pandas de détecter automatiquement
# le séparateur du fichier : virgule, point-virgule, tabulation, etc.
#
# Toutes les colonnes sont lues comme du texte afin de préserver les codes
# et leurs éventuels zéros initiaux.
# =============================================================================

print("\n" + "=" * 72)
print("ÉTAPE 4 — LECTURE DU RÉFÉRENTIEL")
print("=" * 72)

appellations = pd.read_csv(
    chemin_csv,
    sep=None,
    engine="python",
    encoding="utf-8-sig",
    dtype=str,
    keep_default_na=False,
)

# Uniformisation des noms de colonnes :
#   - retrait d’un éventuel marqueur UTF-8 ;
#   - retrait des espaces ;
#   - passage en minuscules.
appellations.columns = [
    str(colonne)
    .replace("\ufeff", "")
    .strip()
    .lower()
    for colonne in appellations.columns
]

print("Colonnes détectées :")
print(appellations.columns.tolist())


# %%
# =============================================================================
# ÉTAPE 5 — NETTOYAGE MINIMAL DES VARIABLES
# =============================================================================
#
# À ce stade, on ne transforme pas encore les libellés pour le calcul des
# proximités. On nettoie seulement les espaces parasites dans les variables
# indispensables.
# =============================================================================

colonnes_a_nettoyer = [
    "code_ogr",
    "code_rome",
    "libelle_appellation_long",
    "classification",
]

for colonne in colonnes_a_nettoyer:
    if colonne in appellations.columns:
        appellations[colonne] = (
            appellations[colonne]
            .astype(str)
            .str.strip()
        )

# La classification est mise en majuscules pour sécuriser les comptages.
if "classification" in appellations.columns:
    appellations["classification"] = (
        appellations["classification"].str.upper()
    )


# %%
# =============================================================================
# ÉTAPE 6 — CONTRÔLE DU SCHÉMA
# =============================================================================
#
# Le traitement ne peut pas continuer si une variable indispensable est
# absente. Cette vérification permet d’arrêter immédiatement le programme
# avec un message compréhensible.
# =============================================================================

print("\n" + "=" * 72)
print("ÉTAPE 6 — CONTRÔLE DU SCHÉMA")
print("=" * 72)

colonnes_obligatoires = {
    "code_ogr",
    "libelle_appellation_long",
    "code_rome",
    "classification",
}

colonnes_absentes = (
    colonnes_obligatoires
    - set(appellations.columns)
)

if colonnes_absentes:
    raise ValueError(
        "Colonnes absentes du référentiel : "
        f"{sorted(colonnes_absentes)}"
    )

print("Toutes les colonnes obligatoires sont présentes.")


# %%
# =============================================================================
# ÉTAPE 7 — CONTRÔLE DES VOLUMES DU ROME 4.61
# =============================================================================
#
# Les contrôles portent sur :
#   - le nombre total de lignes ;
#   - le nombre de codes OGR distincts ;
#   - le nombre de codes ROME distincts ;
#   - le nombre d’appellations principales ;
#   - le nombre d’appellations classées comme synonymes ;
#   - l’absence de doublon sur le code OGR.
# =============================================================================

print("\n" + "=" * 72)
print("ÉTAPE 7 — CONTRÔLE DES VOLUMES")
print("=" * 72)

nb_lignes = len(appellations)
nb_ogr = appellations["code_ogr"].nunique()
nb_rome = appellations["code_rome"].nunique()

repartition_classification = (
    appellations["classification"]
    .value_counts(dropna=False)
)

nb_principales = int(
    repartition_classification.get("PRINCIPALE", 0)
)

nb_synonymes = int(
    repartition_classification.get("SYNONYME", 0)
)

nb_doublons_ogr = int(
    appellations["code_ogr"].duplicated().sum()
)

print("Nombre de lignes :", nb_lignes)
print("Codes OGR distincts :", nb_ogr)
print("Codes ROME distincts :", nb_rome)
print("Appellations principales :", nb_principales)
print("Appellations synonymes :", nb_synonymes)
print("Codes OGR en doublon :", nb_doublons_ogr)

assert nb_lignes == NB_OGR_ATTENDU, (
    f"{NB_OGR_ATTENDU} lignes attendues, "
    f"{nb_lignes} obtenues."
)

assert nb_ogr == NB_OGR_ATTENDU, (
    f"{NB_OGR_ATTENDU} codes OGR attendus, "
    f"{nb_ogr} obtenus."
)

assert nb_rome == NB_ROME_ATTENDU, (
    f"{NB_ROME_ATTENDU} codes ROME attendus, "
    f"{nb_rome} obtenus."
)

assert nb_principales == NB_PRINCIPALES_ATTENDU, (
    f"{NB_PRINCIPALES_ATTENDU} appellations principales attendues, "
    f"{nb_principales} obtenues."
)

assert nb_synonymes == NB_SYNONYMES_ATTENDU, (
    f"{NB_SYNONYMES_ATTENDU} synonymes attendus, "
    f"{nb_synonymes} obtenus."
)

assert nb_doublons_ogr == 0, (
    f"{nb_doublons_ogr} codes OGR en doublon détectés."
)

print("\nContrôle du référentiel ROME 4.61 : OK")


# %%
# =============================================================================
# ÉTAPE 8 — APERÇU DES DONNÉES
# =============================================================================
#
# Cet aperçu est uniquement informatif. Il permet de vérifier visuellement
# que les codes et les libellés ont été lus correctement.
# =============================================================================

print("\n" + "=" * 72)
print("ÉTAPE 8 — APERÇU DU RÉFÉRENTIEL")
print("=" * 72)

colonnes_apercu = [
    "code_ogr",
    "libelle_appellation_long",
    "code_rome",
    "classification",
]

print(
    appellations[colonnes_apercu]
    .head(10)
    .to_string(index=False)
)

print(
    "\nLe référentiel est disponible dans la variable "
    "'appellations'."
)

print(
    "Prochaine étape : préparation et normalisation "
    "des libellés."
)
# %%
