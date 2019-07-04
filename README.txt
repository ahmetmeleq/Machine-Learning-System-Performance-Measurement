This code makes you measure the performance of a similarity-score based recognition system. 

As an example:

- Suppose there is a machine learning system, granting or denying access to a particular room, based on your face photograph. If you are registered before and if it recognizes you, then it grants you permission to get in.

- This machine learning system has previously registered face photographs of various people, including you, inside its database.

- What it does is, it takes your face photograph, generates a key-point feature map from it, and then it compares it with the other faces' features in its database. Then it assigns a similarity point, comparing each face in its database to your faces.

(In the next step our project starts)

- In this repository's project, we have the similarity score matrix of this system. In each element, there is a comparison score, of, say, Photo X and Photo Y. And we know who photo x and photo y belongs to. We can say if the system assigned a good similarity score, based on our knowledge of the people residing in photo x and photo y.

- We look at those similarity scores that the ML system produced and we evaluate the performance of the system, in the best possible scenario. Best possible scenario is when system chooses the lowest-similarity-score-limit-to-grant-access in a wise way.

- We find the wisest selection of the limit, and then we check how many mistakes the system makes, with that similarity score limit.

- The mistakes are expressed with FRR, FAR and EER values; and the ROC curve. You can check their definitions on the internet.



___________________________________________________________________

Hello, this is a guide to work the code.

___________________________________________________________________

TO MAKE THE CODE WORK:

Just run the main.py , there are data files included to test the code.

Generate similarity matrix data that is in a similar form with data data_1_SM and 
generate labels data that is in a similar form with data_1_Class_Labels .

Then you can use the code to evaluate your Machine Learning System.

___________________________________________________________________


REQUIREMENTS

cycler==0.10.0
kiwisolver==1.0.1
matplotlib==3.0.3
numpy==1.16.2
pandas==0.24.1
pyparsing==2.3.1
python-dateutil==2.8.0
pytz==2018.9
scikit-learn==0.20.3
scipy==1.2.1
seaborn==0.9.0
six==1.12.0
sklearn==0.0



___________________________________________________________________

WHAT THE PROGRAM WILL GENERATE?

Program will generate these inside the project folder:

	-log files (.txt) for the processes:
		-Finding EER
		-Finding the FRR value on the point FAR = 0.1

	-genuine impostor plot png file.

	-roc plot

	-a general log file (txt) that is about which processes were completed succesfully. this will also show up on your terminal.


		
___________________________________________________________________


CONTACT

you can contact me here:	ahmet.melek@boun.edu.tr

___________________________________________________________________
