# 📈 Stock Market Prediction using Graph Neural Networks

## 📝 Overview
This project leverages **Graph Neural Networks (GNNs)** to predict stock market trends. It integrates market and news data using a `GraphCNN` architecture for enhanced forecasting accuracy. The model extracts insights from multiple data sources, including historical stock prices, financial news sentiment, and executive relationships.

## 📂 Project Structure

📁 `Model.py` - Implements the `GraphCNN` for stock prediction.  
📁 `Layers.py` - Defines custom graph-based layers for feature extraction.  
📁 `utils.py` - Provides utility functions for preprocessing and evaluation.  
📁 `main1.py` - Entry point for training and evaluating the model.  
📁 `data/` - Contains datasets required for model training.

---

## 🔑 Key Components

### 🚀 GraphCNN (Defined in `Model.py`)
✔ **Graph-based Sequential Modeling** - Uses Graph GRUs for time-series analysis.  
✔ **Attention Mechanism** - Graph attention layers enhance focus on critical relationships.  
✔ **Multi-Relational Graph Construction** - Processes explicit and implicit relationships between stocks and executives.  
✔ **Dropout and Activation Functions** - Uses ELU activation and dropout for better generalization.  

### ⚙️ Custom Layers (`Layers.py`)
🔹 **Graph_GRUModel** - Implements Graph-based GRU networks.  
🔹 **Graph_Tensor** - Merges different features before feeding into GRUs.  
🔹 **Graph_Attention** - Applies attention mechanisms for relational modeling.  
🔹 **SMPLayer** - Performs hierarchical graph convolutions.  
🔹 **Graph_Linear** - Final transformation layer before predictions.  

### 🛠️ Utility Functions (`utils.py`)
📊 **Data Preprocessing** - Reads and processes stock market and news datasets.  
🔗 **Graph Construction** - Creates edge lists and adjacency matrices for model training.  
📈 **Performance Metrics** - Computes evaluation metrics like accuracy and loss.  

---

## 📊 Data & Preprocessing
The dataset comprises:  
✅ **Stock Market Data** - Historical prices, trading volume, and financial indicators.  
✅ **News Data** - Sentiment analysis from financial news.  
✅ **Graph Data** - Relations between stocks and executive teams.  

### 🔄 Data Processing Pipeline (`main1.py`)
- 📥 **Loads stock price data** from CSV files.  
- 📰 **Extracts sentiment scores** from financial news.  
- 🔗 **Builds graph relations** based on company interactions.  
- 📊 **Performs feature normalization** for improved model training.  
- 🚀 **Splits data into training and testing sets** for validation.  

### 🔧 Important Note
💡 If running the model on **CSI300E**, update all dataset paths accordingly in `main1.py`.

### 📁 Dataset Files in `data/CSI100E/`
📂 `x_num_standard.pkl` - Standardized numerical market data.  
📂 `y_1.pkl` - Stock price labels.  
📂 `x_newtext.pkl` - Sentiment data from news articles.  
📂 `edge_new.pkl` - Graph relationships between stocks.  
📂 `interactive.pkl` - Executive team information.  

---

## 🔧 Installation
Ensure dependencies are installed before running the model:
```sh
pip install -r requirements.txt
```

## 🚀 Usage
To train and run the model:
```sh
python main1.py
```

## 🚀 Future Enhancements
✅ Improve attention mechanisms for better feature selection.  
✅ Experiment with advanced graph-based layers.  
✅ Optimize training for large-scale stock datasets.  

## ✍️ Authors
**Saurav Jha, Gaurav Yadav, Ritankar Mukharjee, Rahul Churiwal, Aman Pandey**

