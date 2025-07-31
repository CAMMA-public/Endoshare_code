# Endoshare

This repository contains the Endoshare GUI application.  The original monolithic
`main.py` has been split into a small entry script and a package structure.

```
endoshare/
  app.py            # application bootstrap
  gui/              # Qt widgets and windows
  processing/       # video processing helpers
  resources/        # static data such as icons and models
  utils/            # utility helpers
```

Run the application with:

```bash
python main.py
```
