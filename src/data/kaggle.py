import logging
from pathlib import Path
import zipfile
import kaggle


def download_dataset(
    owner: str,
    name: str,
    dest: Path,
    logger: logging.Logger = logging.getLogger(__name__),
) -> Path:
    dest.mkdir(parents=True, exist_ok=True)

    file_path = dest / f"{name}.zip"

    if file_path.is_file():
        logger.info(
            "Found [ %s ] dataset in [ %s ]. Skipping download...", name, file_path
        )
    else:
        logger.info("Downloading [ %s ] dataset to [ %s ]", name, file_path)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset=f"{owner}/{name}", path=dest)

    return file_path


def download_competition_dataset(
    competition: str,
    dest: Path,
    logger: logging.Logger = logging.getLogger(__name__),
) -> Path:
    dest.mkdir(parents=True, exist_ok=True)

    file_path = dest / f"{competition}.zip"

    if file_path.is_file():
        logger.info(
            "Found [ %s ] dataset in [ %s ]. Skipping download...",
            competition,
            file_path,
        )
    else:
        logger.info("Downloading [ %s ] dataset to [ %s ]", competition, file_path)
        kaggle.api.authenticate()
        kaggle.api.competition_download_files(competition, path=dest)

    return file_path


def unzip_file(
    archive: Path,
    logger: logging.Logger = logging.getLogger(__name__),
) -> Path:
    archive_name = archive.stem
    dest_file = archive.parent / archive_name

    if dest_file.exists():
        logger.info("[ %s ] is already unzipped. Skipping ...", dest_file)
    else:
        logger.info("Unzipping [ %s ] to [ %s ]", archive, dest_file)
        with zipfile.ZipFile(archive, "r") as zip_ref:
            zip_ref.extractall(dest_file)

    return dest_file


def submit_competition(path, message, competition):
    kaggle.api.authenticate()
    kaggle.api.competition_submit(path, message, competition)
