{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4e43ea46-cff0-4c77-8898-0e3d586a5eb8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import plotly.express as px\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import ConfusionMatrixDisplay\n",
    "from sklearn.preprocessing import StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "cfa7309e-3441-46e8-ba7a-992a0076ee60",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-13 02:02:02.948 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Nicholas\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "st.title(\"Airplane Passenger Satisfaction\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "5385ef43-4fe3-41dd-aae8-5d742dd8ae9a",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.write(\"Welcome to the deep insight of passenger satisfaction with their air travel. With so many factors that go into being satisfoed or unsatisfied, it will be so thrilling to see how everything reacts to one another. We will eventually look into making our own predictions with some given information!\") "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1319202e-7fc9-4cc5-880c-b65d2caf83f4",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
