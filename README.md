````markdown
# Can You Break the CAPTCHA?

## 📌 Project Overview
This project consists of **three tasks** focused on CAPTCHA understanding and generation:

- **Task0:** Synthetic data generation  
- **Task1:** Closed-set classification  
- **Task2:** Open-set image-to-text generation  

This repository contains all the code files used to complete the above tasks.

---

## 📂 Repo Structure

```text
PrecogTask/
│── Task0/           # Notebook used for synthetic data generation
│── Task1/           # Scripts, code, and log files for classification task
│── Task2/           # Scripts, code, and log files for generation task
│── README.md        # Project documentation
│── crnn_final.pth   # Saved model for Task2
````

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/kja8586/PrecogTask.git
```

### 2️⃣ Navigate to the Repository

```bash
cd PrecogTask
```

### ▶️ Running Task0 (Synthetic Data Generation)

1. Upload the notebook in **Task0/** to **Google Drive**.
2. Run the **first cell**, then **restart the session** (some downgraded modules are required).
3. For **Hard Set** and **Bonus Set**, upload the **fonts** and **background** folders to Drive and update paths in the notebook.
4. Run all cells to generate all datasets.

### ▶️ Running Task1 and Task2

* If using an **HPC or local server**, replace the scripts with compatible versions and submit.
* Otherwise, run locally:

```bash
python Task1/task1.py
python Task2/task2.py
```

---

## 📝 Notes

* Ensure all required dependencies are installed before execution.
* The trained model for Task2 is provided as `crnn_final.pth`.

---

## 👤 Author

* GitHub: [https://github.com/kja8586](https://github.com/kja8586)
