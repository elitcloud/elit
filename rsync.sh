#!/bin/bash
IP=$1
USER=$2
rsync -avz --delete --exclude-from ".gitignore" /Users/jdchoi/Documents/Software/elit/python -e "ssh -i /Users/jdchoi/Documents/Software/emorynlp.pem" ubuntu@ec2-$IP.compute-1.amazonaws.com:/home/ubuntu/$USER
