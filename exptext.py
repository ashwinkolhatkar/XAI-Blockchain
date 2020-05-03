def generate_text_explanation(per, sample, anchs_vec):
    # -- Assumes -9s are accounted for --
    text_string = "The model predicts that this client is "

    if (per > 0.5):
        text_string += "good"
    else:
        text_string += "bad"

    percentage = per*100.
    text_string += " and has a score of " + str(float(percentage)) + "%\n\n"

    anchs_lst = []

    # -- Create column lists --
    if (anchs_vec is not None):
        for an in range(len(anchs_vec)):
            if (anchs_vec[an] != 0):
                anchs_lst.append(an)
        text_string += anchs_text_exp(sample, anchs_lst, per)

    # print(text_string)
    return text_string




def anchs_text_exp(sample, col_lst, per):

    # -- Writes the header line -- 
    if (per>0.5):
        explanation = "Key Features \nThe key factors that have contributed to the model's positive decision:"
    else:
        explanation = "Key Features \nThe key factors that have contributed to the model's negative decision:"

    for col in col_lst:


        # print(col)

        if (col == 0):
            name = " External Risk Estimate "
            val = str(sample[0])
            # Monotonicity Decreasing
            if (per>0.5):
                explanation += "\n - "+ "The high value for" + name
            else:
                explanation += "\n - "+ "The low value for" + name


        elif (col == 1):
            name = " Months Since Oldest Trade Open"
            # Monotonicity Decreasing
            val = str(sample[1])
            if (per>0.5): 
                explanation += "\n - "+ "A considerable period of " + val + " Months has passed Since Oldest Trade was Open."
            else: 
                explanation += "\n - "+ "Only " + val + " Months have passed Since Oldest Trade was Open."
        
        elif (col == 2):
            name = "Months Since Last Trade Open"
            # Monotonicity Decreasing
            val = str(sample[2])
            if (per>0.5):
                explanation += "\n - "+ "A considerable period of " + val + " Months have passed Since a Recent Trade was Open."
            else:
                explanation += "\n - "+ "Only " + val + " Months have passed Since a Recent Trade was Open."
        
        elif (col == 3):
            name = "Average Months in File"
            # Monotonicity Decreasing
            val = str(sample[3])
            if (per>0.5):
                explanation += "\n - "+ "The client has a high average of " + val + " Months in File."
            else:
                explanation += "\n - "+ "The client has only an average of " + val + " Months in File."

        elif (col == 4):
            name = "Satisfactory Trades"
            # Monotonicity Decreasing
            val = str(sample[4])
            if (per>0.5):
                explanation += "\n - "+ "The client has a high number of " + name 
            else:
                explanation += "\n - "+ "The client has a low number of " + name

        elif (col == 5):
            name = "Trades 60+ Ever"
            # Monotonicity Increasing
            val = str(sample[5])
            if (per>0.5):
                explanation += "\n - "+ "The client has very few trade lines for which payment was received 60 days past its due date"
            else:
                explanation += "\n - "+ "The client has too many trade lines for which payment was received 60 days past its due date"

            # WHAT IS THIS??!?
        elif (col == 6):
            name = "Trades 90+ Ever"
            # Monotonicity Increasing
            val = str(sample[6])
            if (per>0.5):
                explanation += "\n - "+ "The client has very few trade lines for which payment was received 90 days past its due date"
            else:
                explanation += "\n - "+ "The client has too many trade lines for which payment was received 90 days past its due date"
        elif (col == 7):
            name = "% Trades Never Delq."
            # Monotonicity Decreasing
            val = str(sample[7])
            if (per>0.5):
                explanation += "\n - "+ "This client has a overwhelming percentage of Trades that have Never gone Delinquent."
            else:
                explanation += "\n - "+ "This client has a alarmingly small percentage of Trades that have Never gone Delinquent."

        elif (col == 8):
            name = "Months Since Most Recent Delq."
            # Monotonicity Decreasing
            val = str(sample[8])
            if val == "150":
                explanation += "\n - "+ "This person has never been delinquent"
            else:
                if (per>0.5):
                    explanation += "\n - "+ val + " months have passed since Last Deliquency."
                else:
                    explanation += "\n - "+ "Only " + val + " Months have passed since Last Deliquency."
        elif (col == 9):
            name = "Max Delq. Last 12M"
            # Monotonicity Decreasing
            val = sample[9]
            if (val == 0):
                explanation += "\n - "+ "The Maximum Deliquency in the last 12 Months for this client does not exist."
            elif (val == 1):
                explanation += "\n - "+ "The Maximum Deliquency in the last 12 Months for this client is unknown."
            elif (val == 2):
                explanation += "\n - "+ "The Maximum Deliquency in the last 12 Months indicates a derogatory comment."
            elif (val == 3):
                explanation += "\n - "+ "The Maximum Deliquency in the last 12 Months for this client is over 120 days."
            elif (val == 4):
                explanation += "\n - "+ "The Maximum Deliquency in the last 12 Months for this client is over 90 days."
            elif (val == 5):
                explanation += "\n - "+ "The Maximum Deliquency in the last 12 Months for this client is over 60 days."
            elif (val == 6):
                explanation += "\n - "+ "The Maximum Deliquency in the last 12 Months for this client is over 30 days."
            elif (val == 7):
                explanation += "\n - "+ "This client has not been delinqent in the past 12 months."

     
        elif (col == 10):
            name = "Max Delq. Ever"
            # Monotonicity Decreasing
            val = sample[10]

            if (val == 0):
                explanation += "\n - "+ "The Maximum Deliquency Ever for this client does not exist."
            elif (val == 1):
                explanation += "\n - "+ "The Maximum Deliquency Ever for this client is unknown."
            elif (val == 2):
                explanation += "\n - "+ "The Maximum Deliquency Ever indicates a derogatory comment."
            elif (val == 3):
                explanation += "\n - "+ "The Maximum Deliquency Ever for this client is over 120 days."
            elif (val == 4):
                explanation += "\n - "+ "The Maximum Deliquency Ever for this client is over 90 days."
            elif (val == 5):
                explanation += "\n - "+ "The Maximum Deliquency Ever for this client is over 60 days."
            elif (val == 6):
                explanation += "\n - "+ "The Maximum Deliquency Ever for this client is over 30 days."
            elif (val == 7):
                explanation += "\n - "+ "This client has never been delinqent."

        elif (col == 11):
            name = "Total Trades"
            # Monotonicity Unknown
            val = str(sample[11])
            if (per>0.5):
                explanation += "\n - "+ "The number of " + name + "is an important factor in the positive decision."
            else:
                explanation += "\n - "+ "The number of " + name + "is an important factor in the negative decision."
        elif (col == 12):
            name = "Trades Open Last 12M"
            # Monotonicity Increasing
            val = str(sample[12])
            if (per>0.5):
                explanation += "\n - "+ "The client has a Opened a very small number of trades in the past 12 Months."
            else:
                explanation += "\n - "+ "The client has a Opened a high number of trades in the past 12 Months."

        elif (col == 13):
            name = "% Installment Trades"
            # Monotonicity Unknown
            val = str(sample[13])
            explanation += "\n - "+ "The percentage of Installment Trades is an important feature for the model's decision"

        elif (col == 14):
            name = "Months Since Most Recent Inq"
            # Monotonicity Decreasing
            val = str(sample[14])
            if val == "150":
                explanation += "\n - "+ "No inquiries have been made against this person."
            if (per>0.5): 
                explanation += "\n - "+ val + " Months have passed since the most Recent Inquiry"
            else:
                explanation += "\n - "+ "Only " + val + " months have passed since the most Recent Inquiry"
        elif (col == 15):
            name = "Inq Last 6 Months"
            # Monotonicity Increasing
            val = str(sample[15])
            if (per>0.5):
                explanation += "\n - "+ "The client has only " + val + "inquiries in the Past 6 Months"
            else:
                explanation += "\n - "+ "The client has too many inquiries in the Past 6 Months"
        elif (col == 16):
            name = "Inq Last 6 Months exl. 7 days"
            # Monotonicity Increasing
            val = str(sample[16])
            if (per>0.5):
                explanation += "\n - "+ "The client has only " + val + "inquiries in the Past 6 Months when excluding the last 7 days"
            else:
                explanation += "\n - "+ "The client has too many inquiries in the Past 6 Months even when excluding the last 7 days"
        elif (col == 17):
            # print("enter")
            name =  "Revolving Burden"
            # Monotonicity Increasing
            val = str(sample[17])
            if (per>0.5):
                explanation += "\n - "+ "The client has a good ratio of revolving balance against their credit limit." 
            else:
                explanation += "\n - "+ "The client has a poor ratio of revolving balance against their credit limit." 
        elif (col == 18):
            name =  "Installment Burden"
            # Monotonicity Increasing
            val = str(sample[18])
            if (per>0.5):
                explanation += "\n - "+ "The client has maintained a good ratio for installment balance against their original loan amount." 
            else:
                explanation += "\n - "+ "The client has averaged a poor ratio for installment balance against their original loan amount." 
        elif (col == 19):
            name =  "Revolving Trades w/ Balance"
            # Monotonicity Unknown
            val = str(sample[19])
            explanation += "\n - "+ "The number of Revolving Trades with balance is " + val 
                   
        elif (col == 20):
            name =  "Installment Trades w/ Balance"
            # Monotonicity Unknown
            val = str(sample[20])
            explanation += "\n - "+ "The number of Revolving Trades with balance is " + val

        elif (col == 21):
            name =  "Bank Trades w/ High Utilization Ratio"
            # Monotonicity Increasing
            val = str(sample[21])
            if (per>0.5):
                explanation += "\n - "+ "Only a small number of Bank Trades have a High Utilization Ratio"
            else:
                explanation += "\n - "+ "Too many Bank Trades have a High Utilization Ratio"
        elif (col == 22):
            name = "% trades with balance"
            # Monotonicity Unknown
            val = str(sample[22])
            explanation += "\n - "+ "Maintained "+ val +" percentage of trades with balance"

    return explanation
