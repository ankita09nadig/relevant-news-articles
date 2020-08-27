Pre-Requisite: 
	Python 3.0 or above

To setup the project follow these steps:

1) Install MySQL on your platform and setup root password while installing mysql.
2) Open demoapp/db.yaml file in any text editor and change the MySQL username and password 
   attributes that match the system.
3) Open app.py and please install the following dependencies.
   Import libraries: (i) flask_mysqldb		(vii) math
		     (ii) PyYAML and yaml	(viii) MySQl
		     (iii) numpy		(ix) string
		     (iv) nltk			(x) codecs
		     (v) natsort		(xi) os
		     (vi) hashlib		(xii) flask
4) Change the path_file variable in line 409 of app.py so that it reaches the demoapp directory.
5) Open MySQL and create a database using the following commands: 
	Create database dbname;
	use dbname;
	Create table login(id INT NOT NULL AUTO_INCREMENT, email VARCHAR(100) NOT NULL, password VARCHAR(40) NOT NULL, PRIMARY KEY (id));				     
6) Open a git bash in the 'mews' directory of the project folder
7) Start a virtual environment by executing the command "source venv/bin/activate"
8) Then change directory to demoapp and in the bash execute the command "flask run"
8) The application will start on http://localhost:5000
9) Go to any browser and open this URL
10) Create an account and login using email and password
