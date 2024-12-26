import os

def fetch_instruction_content(prompt_identifier):
    txt_ext = ".txt"
    target = f"{prompt_identifier}{txt_ext}"

    locations = [
        (os.path.join(os.path.dirname(__file__), "system"), False),
        (os.path.join(LABO_DIR, "system_prompts"), True),
    ]

    for storage_path, create_if_missing in locations:
        if create_if_missing and not os.path.exists(storage_path):
            try:
                os.makedirs(storage_path)
            except OSError:
                continue

        content_path = os.path.join(storage_path, target)
        try:
            with open(content_path, "r", encoding="utf-8") as src:
                return src.read().strip()
        except (IOError, FileNotFoundError):
            continue

    raise FileNotFoundError(
        f"Instruction data unavailable - id: {prompt_identifier}, "
        f"last attempted: {content_path}"
    )
