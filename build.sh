git pull
# DEFINE PATH
dir1=${PWD}

# START FRESH
rm -rf _site;

# BUILD WEBSITE
quarto render

# CLEAN UP 
cd _site; for i in $(find  ./ -name .DS_Store); do rm $i; done; cd "$dir1"

# SET CORRECT PERMISSIONS FOR ALL FILES 
for i in $(find _site -type f); do chmod 644 $i; done
for i in $(find _site -type d); do chmod 755 $i; done


# PUSH WEBSITE TO GU DOMAINS 
printf 'Would you like to push to GU domains? (y/n)? '
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then 
    rsync -alvr --delete _site/* normanwg@gtown03.reclaimhosting.com:/home/normanwg/public_html/nnproject/
else
    echo NOT PUSHING TO GU DOMAINS!
fi

# GITHUB SYNC
printf 'Would you like to push to GITHUB? (y/n)? '
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then 

    git config http.postBuffer 20242880000

    # PULL CLOUD REPO TO LOCAL
    git pull 

    # SYNC TO LOCAL REPO TO CLOUD 
    read -p 'ENTER MESSAGE: ' message
    echo "commit message = "$message; 
    git add ./; 
    # MAIN BRANCH
    git commit -m "$message"; 

    # PUSH NON-MAIN BRANCH
    #git push  -u origin w05-draft

    # PUSH MAIN BRANCH
    git push

else
    echo NOT PUSHING TO GTIHUB!
fi







# PUSH WEBSITE TO GU DOMAINS 
# printf 'Would you like to push to GU domains? (y/n)? '
# read answer
# if [ "$answer" != "${answer#[Yy]}" ] ;then 
#     rsync -alvr --delete _site/* jfhgeorg@gtown.reclaimhosting.com:/home/jfhgeorg/public_html/dsan-5000/
# else
#     echo NOT PUSHING TO GU DOMAINS!
# fi