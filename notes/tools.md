`ssh skivelso@sherlock.stanford.edu`

VSCode shortcuts:

`cmd+shift+x` explorer

`cmd+shift+p` search functionalities

`cmd+p` search files

`cmd+shift+e` files 

Connect to Sherlock in VSCode:

`cmd+shift+p` then "SSH FS: Connect..."

Git commands

`git clone https://github.com/geoffreyangus/analytic-continuation.git`

```
git status
***check what you did***
git add .
git commit -am "add your message here"
git push
```

change `data/data_06-19-19/params.json` in Sophia's repo in order to have proper permissions

```
source /home/users/gangus/analytic-continuation/.env/bin/activate
ml python/3.6
ml py-pytorch/1.0.0_py36
ml py-numpy/1.14.3_py36
```

running an experiment

```
***confirm you've updated submit.sh with the right params.json file***
sbatch submit.sh
squeue -u gangus // to double check the job status
```

canceling an experiment

```
squeue -u gangus
***On the left you will see the JOB_ID***
scancel <JOB_ID>
```

