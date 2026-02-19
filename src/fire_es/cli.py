"""
CLI для проекта Fire ES.

Команды:
- notebooks: исполнение ноутбуков с сохранением результатов
"""

import subprocess
import sys
from pathlib import Path

import click


@click.group()
def cli():
    """Fire ES CLI - управление проектом."""
    pass


@cli.command()
@click.option(
    "--notebook",
    "-n",
    multiple=True,
    help="Имя ноутбука для исполнения (можно несколько)",
)
@click.option(
    "--output-dir",
    "-o",
    default="reports",
    help="Директория для сохранения результатов",
)
def notebooks(notebook: tuple[str, ...], output_dir: str):
    """
    Исполнение ноутбуков с сохранением результатов.

    Если не указаны конкретные ноутбуки, исполняются все в notebooks/.
    """
    notebooks_dir = Path("notebooks")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Если не указаны ноутбуки, берём все
    if not notebook:
        notebooks_list = sorted(notebooks_dir.glob("*.ipynb"))
        if not notebooks_list:
            click.echo("Нет ноутбуков в директории notebooks/")
            return
    else:
        notebooks_list = [notebooks_dir / name for name in notebook]

    click.echo(f"Исполнение {len(notebooks_list)} ноутбуков...")

    for nb_path in notebooks_list:
        if not nb_path.exists():
            click.echo(f"⚠ Ноутбук не найден: {nb_path}")
            continue

        click.echo(f"📓 Исполнение: {nb_path.name}")

        # Используем papermill для исполнения
        output_nb = output_path / nb_path.name
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "papermill",
                    str(nb_path),
                    str(output_nb),
                    "--log-output",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            click.echo(f"✅ Успешно: {output_nb}")
        except subprocess.CalledProcessError as e:
            click.echo(f"❌ Ошибка: {nb_path.name}")
            click.echo(e.stderr)
        except FileNotFoundError:
            click.echo("⚠ papermill не установлен. Установите: pip install papermill")
            # Пробуем через nbconvert
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "nbconvert",
                        "--execute",
                        "--to",
                        "notebook",
                        "--output",
                        str(output_path),
                        str(nb_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                click.echo(f"✅ Успешно (nbconvert): {output_nb}")
            except subprocess.CalledProcessError as e2:
                click.echo(f"❌ Ошибка (nbconvert): {nb_path.name}")
                click.echo(e2.stderr)


@cli.command()
def version():
    """Показать версию пакета."""
    from fire_es import __version__

    click.echo(f"Fire ES v{__version__}")


if __name__ == "__main__":
    cli()
