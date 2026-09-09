This model is useful to train the yolo API
We have different classes to fetch the information we required in this case basketball players and balls
After retrieving the info a images folder is created and it was required to be normalized with flatten_dataset
Having all the images to train YOLO we need to obay a certain folder logic

dataset /
images / train / val
label / train / val

For this we manually create the dataset 
We need to have the images and labels folder outsite the DATASET FOLDER
we will require to run python3 ../autolabel.py to generate the labels
and run python3 ../split_dataset.py to generate train and val folders
once completed lets train the data by run python3 ../train.py

-- This process is long 
The model will be stored in best and will learn continously by always picking up best from last iterationGET


// BEFORE NEXT TRAINING

python3 -c "import torch; print(torch.backends.mps.is_available())"
IF TRUE ready to use mps if false fix pyton torch