According to Federico's dataset workflow understanding :), 
these will be the steps to perform:

1. Put in folder Images the original (phase1 images) pluts the new ones from phase2.

2. Run the original webeep python script to generate the old dataset. It will be equal to phase1 df_old in the upper part, then there is the stuff about the new images.

3. This will populate the Processed dataset folder. Again, the upper part of the images and dataset should be equal to our df_new.

4. From that we can build the charts using the upper (0->39 rows) part of the data, and the subsequent rows as test data.