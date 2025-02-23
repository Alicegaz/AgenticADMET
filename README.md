## Install Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install -r requirements.txt --upgrade pip --link-mode=copy
```


### export your venv
```bash
uv pip freeze > requirements_pre.txt
```
