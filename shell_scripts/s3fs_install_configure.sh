#Uncomment the lines appropriate for your OS (Ignore if using Windows. Use NFS)

#On Debian 9 and Ubuntu 16.04 or newer:

sudo apt-get install s3fs

#On SUSE 12 or newer and openSUSE 42.1 or newer:

sudo zypper in s3fs

#On Fedora 27 and newer:

sudo yum install s3fs-fuse

#On RHEL/CentOS 7 and newer through EPEL repositories:

sudo yum install epel-release
sudo yum install s3fs-fuse

#On Amazon Linux through EPEL repositories:

sudo amazon-linux-extras install epel
sudo yum install s3fs-fuse

#On macOS, install via Homebrew:

brew cask install osxfuse
brew install s3fs

# Save credentials
echo ACCESS_KEY_ID:SECRET_ACCESS_KEY > ${HOME}/.passwd-s3fs
chmod 600 ${HOME}/.passwd-s3fs



