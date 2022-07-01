# Analysis of 4dm4 scores

### Pending

- [ ] Data Cleaning / Engineering

  - [ ] Pulling and Reformatting 4dm3 Data into SQLite Database
  - [ ] Cleaning Ranked Plays dataset

- [ ] Analysis

  - [ ] Evaluate Models

- [ ] Questionnaire

  - [ ] Create Questions
  - [ ] Create a form
  - [ ] Sheet Engineering
  - [ ] User Form EDA

- [ ] Documentation

  - [ ] Code Refactoring & Documentation
    - [ ] Moving analysis to `.ipynb` if possible (Engineering part is still `.py`)
    - [ ] Interpreting the Results of Outlier Models
    - [ ] Cleaning TODO.md

### In-Progress

- [ ] Data Cleaning / Engineering

  - Nothing to do now

- [ ] Analysis

  - [ ] Outlier Detection Analysis
    - [x] Outlier Detection Analysis with Adjusted LOF
    - [ ] Outlier Detection Analysis with Weighted Percent Models (both Parametric and Non-Parametric)

- [ ] Questionnaire

  - [ ] TBA

- [ ] Documentation

  - [ ] TBA

### Done ✓

- [x] Data Cleaning / Engineering

  - [x] Fetch Data from Excel to one huge chunk of CSV
  - [x] Import them to sqlite3 db
  - [x] Validate the missing Data (KNN Impute / Collaborative Filtering)

- [ ] Analysis

  - [x] Basic EDA
  - [x] Regression Analysis
    - Linear Regression
    - Polynomial Regression Degree 2 and 3
    - Applying L1 and L2 Regularization (Lasso, Ridge)
  - [x] Qualifiers Analysis
    - [x] Survival Analysis
    - [x] Logistic Regression (@poly)
    - [x] Overall Conclusion

- [ ] Questionnaire

  - [ ] TBA

- [ ] Documentation

  - [ ] Methodology
    - [x] EDA Methodology
      - Missing Data Validation
      - Regression Analysis
    - [x] EDA Result
      - Hypothesis Testing
      - Interpretation of Regression Analysis
      - More questions
    - [x] Outlier Detection using LOF
      - Model Interpretation

### External Questions for Further Research

- Do qualifiers seeds actually matter ? Is there any relation between qualifiers seeds and the overall performance in the actual match ?
