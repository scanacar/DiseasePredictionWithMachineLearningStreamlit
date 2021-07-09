You can find detail of project at Jupyter Notebook file

First of all, you need to import the libraries necessary for the Web App to work.

pip install streamlit
pip install scikit-learn
pip install numpy
pip install pandas
pip install pickle

The files are extracted from zip as folders and opened with the IDE.

SDSP_model_building.py file is run first.(Instead of constantly running the 
					model, a dummy pickle is created.)

Write "streamlit run SDSP_app.py" to the terminal part of the IDE you are using.
(The app is executed.)


Text_input was put for float-int values. Its default value is set to 0.
A selectbox has been put for strings and objects. Default value is set as 
the first element of selectbox.

After the values are entered, the Predict button is clicked. The 3 diseases 
with the highest probability (max 3) are shown as the result.

Selectboxes and text_inputs are changed to default value when the Clear button 
is pressed. Also results are deleted.

After making changes in the model of our program, SDSP_model_building.py is run again. 
If there is a decrease or increase in the features, the selectbox or text_input 
decreases or increases dynamically.

Youtube Link : https://www.youtube.com/watch?v=f1jJx7Wse6k&t=18s
