mkdir data/training
s3fs s3://gsv-aerial-imagery data/training -o passwd_file=${HOME}/.passwd-s3fs