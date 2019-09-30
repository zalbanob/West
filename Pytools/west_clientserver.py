#!/usr/bin/python3

from time import sleep, perf_counter as pc
from os import path, remove
from abc import ABC, abstractmethod

##############
# SUPERCLASS #
##############

class ClientServer(ABC): 
   #
   def __init__(self,client_lockfile,maxsec=21600,sleepsec=10):
      #
      self.client_lockfile = client_lockfile
      self.maxsec = maxsec
      self.sleepsec = sleepsec
      super().__init__()
   #
   @abstractmethod
   def before_sleep(self): # subclass needs to implement this method
      pass
   #
   @abstractmethod
   def after_sleep(self): # subclass needs to implement this method
      pass
   #
   @abstractmethod
   def awake_condition(self): # subclass needs to implement this method
      pass
   #
   def start(self):
      #
      # ====================================
      self.before_sleep()
      # ====================================
      #
      awake = 1 # I am awake if this is zero
      t0 = pc()
      while (pc()-t0 <= self.maxsec) :
         #
         # =================================
         self.awake_condition()
         # =================================
         #
         exists = path.exists(self.client_lockfile)
         if (not exists) : 
            awake = 0 
            break
         else : 
            sleep(self.sleepsec)
      #
      # ====================================
      self.after_sleep()
      # ====================================
      return awake

###############
# SERVERCLASS #
###############

class QboxServer(ClientServer) : 
   #
   def before_sleep(self):
       #
       # Determine the name of the server file 
       #
       client_image = self.client_lockfile.split(".")[1] 
       self.server_inputfile = f"qb.{client_image}.in" # we assume that server_number = client_image
       #
       # List of perturbation files 
       #
       perturbation_list = []
       with open(self.client_lockfile,"r") as f:
          for cnt, line in enumerate(f):
             perturbation_list.append(line.replace("\n",""))
       #
       # Create the input file for the server 
       # 
       with open(self.server_inputfile,"w") as f: 
          f.write("load gs.xml\n")
          f.write("set xc PBE\n")
          f.write("set wf_dyn PSDA\n")
          f.write("set scf_tol 1.e-8\n")
          for pert in perturbation_list : 
              f.write(f"response -vext {pert} -IPA -amplitude 0 20\n")
       #
       # Awake server, by removing its lockfile 
       #
       if(path.exists(self.server_inputfile+".lock")) :
          remove(self.server_inputfile+".lock")

   #
   def awake_condition(self): 
       #
       # If server gets to sleeps, awake the client 
       #
       if( path.exists(self.server_inputfile+".lock")) :  
          remove(self.client_lockfile)
   #
   def after_sleep(self):
       pass 

#############
# INTERFACE #
#############

def sleep_and_wait(*args, **kwargs):
    #
    client_lockfile = args[0] # name of client lockfile 
    maxsec = 12 * 60 * 60 # 12 hours, Max sleep time (in s) 
    sleepsec = 1 # 1 second, Sleep interval (in s)
    #
    # change defaults 
    #
    if "maxsec" in kwargs.keys() : 
       maxsec = kwargs["maxsec"]
    if "sleepsec" in kwargs.keys() : 
       sleepsec = kwargs["sleepsec"]
    #
    server = QboxServer(client_lockfile,maxsec,sleepsec)
    return_int = server.start()
    #
    return return_int
    
########
# TEST #
########

def test() :
    with open("I.1.lock","w") as f :
       f.write(" ")
    sleep_and_wait("I.1.lock",maxsec=60,sleepsec=2)

if __name__ == "__main__":
    # execute only if run as a script
    test()

