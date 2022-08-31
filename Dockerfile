#base image from dockerhub .the base image is python 3.8 ,debian based , 
#slim because its optimised
FROM python:3.8.13-slim

#install pipenv on the image
RUN pip install pipenv

#create an 'app' directory is if it doesnt exist and cd there so that its the working directory
WORKDIR /app

#copy the 'pipfile' ,'pipfile.lock' files into the present directory i.e 'app'
COPY ["Pipfile", "Pipfile.lock", "./"]

#install all libary in pipfile.lock file but do not create virtual environment
RUN pipenv install --system --deploy

#copy 'main.py' ,"churn-models.bin", "churn.jpeg" files into working directory i.e 'app' 
COPY ["main.py", "churn-models.bin", "churn.jpeg", "./"]

#used to download package information from all configured sources
#“apt-get update” updates the package sources list to get the latest list of available
# packages in the repositories 
# “apt-get upgrade” updates all the packages presently installed in our 
#Linux system to their latest versions.
RUN apt-get update

#during test..i got an error
#OSError: libgomp.so.1: cannot open shared object file: No such file or directory
RUN  apt-get install libgomp1


#expose port '8501' of the container to the local system/machine
EXPOSE 8501

#entrypoint is the default command that is executed when we do docker run
#>> streamlit run app.py
ENTRYPOINT ["streamlit", "run"]

# run app.py with gunicorn
CMD ["main.py"]
