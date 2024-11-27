

## Quick Setup

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Run setup:
```bash
make all
```

This will:
- Create a Python venv
- Set up PYTHONPATH
- Download training data

3. Activate the venv:
```bash
source venv/bin/activate
```

## Requirements

- Python 3.8 or higher
- make
- unzip


To clean and restart setup:
```bash
make clean
make all
```