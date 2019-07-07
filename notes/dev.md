## 6/21

We don't have write permissions to the $GROUP_SCRATCH folder but we've sent an email to make it happen.

Also the empty file situation seems to be resolved, per Ilya's last email.

#### Running experiments

When you first log on, type the following commands:

```
source .env/bin/activate
ml python/3.6
ml py-pytorch/1.0.0_py36
ml py-numpy/1.14.3_py36
```

#### Malformed Input

Seems to be a syntax error in some of the files. Comes from scientific notation not porting over well to .txt files.

```
/scratch/groups/simes/analytic_cont+machine_learning/data_06-19-19/tmp/lambda_beta32_1558.dat
```

#### Dedicated Login Node

For `screen` capabilities:

