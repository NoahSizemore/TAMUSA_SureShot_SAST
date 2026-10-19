<div align="center">

# NAME: TAMUSA Software Engineering I Term Project

![Project](https://img.shields.io/badge/Project-TAMUSA-blue)

[Noah Sizemore](https://github.com/NoahSizemore) &nbsp;•&nbsp; [David Hinojosa]() &nbsp;•&nbsp; [William Hale](https://github.com/williamhale210-ai)

</div>

## 📝 Project Scope

`TEMPORARY`

- As a team, develop a software application through the entire development life cycle. This means performing requirements analysis, design, implementing a functioning prototype, testing, and documenting the system.
- To develop a tool that helps programmers analyze source code and identify security vulnerabilities.
- Target programming language: **Rust**
    
    If your team wants to choose a different programming language, you need approval from the instructor.
    
- Each team can suggest any tool that helps programmers understand the program
- Topics Examples
    - Static Software Vulnerability Detector
    - Dynamic Software Vulnerability Detector
    - Source Code Visualizer
    - Data Flow Generator/Taint Tracking
    - Refactoring Tool
    - etc

## 📖 About The Project



<div align="center">
  <img src="https://www.vectorstock.com/royalty-free-vector/tbd-outline-icon-to-be-determined-note-minimal-vector-57615992" width="700" height="650">
  <br>
</div>


## ✨ Key Features

### How to use the model

To run the model, simply run the following:
```bash
cd surshot
run_model.sh
```

Alternatively, you could do the following commmands if you wish to no use the script:

```bash
cd sureshot
.venv\Scripts\activate

# train + evaluate + save, in one run
python scripts/train_XGBoost.py \
    --data data/processed/rustsec_v0.1.parquet \
    --source-type rustsec --seed 42 \
    --report reports/mine.json \
    --save artifacts/xgboost

# same for the baseline
python scripts/train_scikit-learn.py \
    --data data/processed/rustsec_v0.1.parquet \
    --source-type rustsec --seed 42 \
    --report reports/reference.json --save artifacts/reference

# side-by-side the two reports
python -c "from sureshot.pipeline import compare; compare('reports/reference.json','reports/mine.json')"
```
