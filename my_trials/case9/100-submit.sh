today=$(date)
kaggle competitions submit -c talkingdata-adtracking-fraud-detection -f $1 -m "From API ${today}"
