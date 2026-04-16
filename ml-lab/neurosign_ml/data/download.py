"""Download WLASL dataset from Kaggle.

Requisitos: 6.1
"""

from pathlib import Path


def download_wlasl(dest_dir: Path) -> Path:
    """Download the WLASL-processed dataset from Kaggle.
    
    Usa a autenticação automática configurada no ~/.kaggle/kaggle.json.

    Args:
        dest_dir: Directory where the dataset will be downloaded and unzipped.

    Returns:
        The destination directory path.
    """
    import kaggle  # noqa: PLC0415

    # Cria o diretório se não existir
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Iniciando download do WLASL para: {dest_dir.absolute()}...")
    
    kaggle.api.dataset_download_files(
        "risangbaskoro/wlasl-processed",
        path=str(dest_dir),
        unzip=True,
    )

    print("✅ Download e extração concluídos com sucesso!")
    return dest_dir

# O "Gatilho": Só executa a função se o arquivo for rodado diretamente pelo terminal
if __name__ == "__main__":
    # Define a pasta destino para salvar os dados (ex: ml-lab/data/raw)
    target_path = Path("data/raw")
    download_wlasl(target_path)