# California Housing Price Prediction - Machine Learning Project

## 📋 Project Overview

This is my first end-to-end machine learning project that predicts median house values in California districts based on various features. The project follows a complete machine learning workflow from data acquisition to model deployment.

## 🎯 Project Goal

To build a predictive model that can accurately estimate median house prices in California based on features like location, income, population, and housing characteristics.

## 📊 Dataset

The dataset contains information from the 1990 California census with the following features:

### Features:
- **longitude**: Longitudinal coordinate
- **latitude**: Latitudinal coordinate  
- **housing_median_age**: Median age of houses in the district
- **total_rooms**: Total number of rooms in the district
- **total_bedrooms**: Total number of bedrooms in the district
- **population**: Population in the district
- **households**: Number of households in the district
- **median_income**: Median income of households (in tens of thousands)
- **ocean_proximity**: Proximity to the ocean (categorical)

### Target Variable:
- **median_house_value**: Median house value for the district (in USD)

## 🛠️ Technologies Used

- **Python 3.7+**
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms and utilities
- **Matplotlib** - Data visualization
- **Joblib** - Model serialization
- **SciPy** - Statistical functions

## 📁 Project Structure

```
california-housing-project/
│
├── california_housing_project.py  # Main project file
├── requirements.txt               # Dependencies
├── california_housing_model.pkl   # Trained model (generated)
├── datasets/                      # Data directory (auto-created)
│   └── housing/                   # Housing dataset
│       └── housing.csv           # Raw data
└── README.md                     # This file
```

## 🚀 Installation & Setup

1. **Clone or download the project files**
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the project**:
   ```bash
   python california_housing_project.py
   ```

## ⚙️ How It Works

### 1. Data Loading & Exploration
- Automatically downloads the dataset if not present
- Performs exploratory data analysis
- Displays data statistics and distributions

### 2. Data Preprocessing
- Handles missing values using median imputation
- Scales numerical features using StandardScaler
- Encodes categorical variables using OneHotEncoder
- Creates stratified train-test split

### 3. Model Training
- Uses Random Forest Regressor as the main algorithm
- Implements a complete pipeline for preprocessing and modeling
- Trains on 80% of the data (16,512 samples)

### 4. Model Evaluation
- Tests on 20% of the data (4,128 samples)
- Calculates Root Mean Squared Error (RMSE)
- Saves the trained model for future use

### 5. Model Deployment
- Serializes the trained model using Joblib
- Demonstrates loading and using the saved model

## 📈 Results

The model achieves a competitive RMSE (Root Mean Squared Error) on the test set, demonstrating good predictive performance for housing price estimation.

## 🎓 Key Learnings

This project helped me understand:

- **End-to-end ML workflow**: From data acquisition to model deployment
- **Data preprocessing**: Handling missing values, scaling, encoding
- **Model selection**: Choosing appropriate algorithms for regression tasks
- **Evaluation metrics**: Using RMSE for regression problems
- **Model persistence**: Saving and loading trained models
- **Best practices**: Code organization and documentation

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- [ ] Feature engineering (creating new features like rooms per household)
- [ ] Experiment with different algorithms (XGBoost, Neural Networks)
- [ ] Create a web interface for predictions
- [ ] Add more visualization and analysis
- [ ] Implement model monitoring and retraining

## 📚 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [California Housing Dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

## 👨‍💻 Author

**Aditya Kulkarni**  
- First machine learning project
- Learning data science and ML fundamentals
- Open to feedback and collaboration

## 📄 License

This project is for educational purposes. Feel free to use and modify as needed.

---

**Note**: This is my first machine learning project! I'm excited to continue learning and improving my skills in data science and machine learning. Feedback and suggestions are always welcome! 🚀
