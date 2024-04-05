import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.corpus import reuters
from nltk.corpus import brown
from nltk.corpus import gutenberg
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import joblib
from collections import Counter
# from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from PIL import Image
import urllib
import requests
import re
from textblob import Word
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from summarization import summarize


def classifyModel():
    df = pd.read_csv("terms&conditions_dataset2.csv",
                     engine='python', encoding='UTF-8')
    df['category'].value_counts()
    df.to_csv("terms&conditions_dataset1.csv", index=False)
    df['category'].value_counts()
    df['text'] = df['text'].fillna("")
    df.isna().sum()
    # preprocess
    df['lower_case'] = df['text'].apply(
        lambda x: x.lower().strip().replace('\n', ' ').replace('\r', ' '))
    df['alphabatic'] = df['lower_case'].apply(lambda x: re.sub(
        r'[^a-zA-Z\']', ' ', x)).apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
    df['without-link'] = df['alphabatic'].apply(
        lambda x: re.sub(r'http\S+', '', x))
    tokenizer = RegexpTokenizer(r'\w+')
    df['Special_word'] = df.apply(
        lambda row: tokenizer.tokenize(row['lower_case']), axis=1)
    stop = [word for word in stopwords.words('english') if word not in ["my", "haven't", "aren't", "can", "no", "why", "through", "herself", "she", "he", "himself", "you", "you're", "myself", "not",
                                                                        "here", "some", "do", "does", "did", "will", "don't", "doesn't", "didn't", "won't", "should", "should've", "couldn't", "mightn't", "mustn't", "shouldn't", "hadn't", "wasn't", "wouldn't"]]
    df['stop_words'] = df['Special_word'].apply(
        lambda x: [item for item in x if item not in stop])
    df['stop_words'] = df['stop_words'].astype('str')
    df['short_word'] = df['stop_words'].str.findall('\\w{2,}')
    df['string'] = df['short_word'].str.join(' ')
    df['Text'] = df['string'].apply(lambda x: " ".join(
        [Word(word).lemmatize() for word in x.split()]))
    # split train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        df["Text"], df["category"], test_size=0.25, random_state=42)
    count_vect = CountVectorizer(ngram_range=(1, 2))
    transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
    x_train_counts = count_vect.fit_transform(x_train)
    x_train_tfidf = transformer.fit_transform(x_train_counts)
    x_test_counts = count_vect.transform(x_test)
    x_test_tfidf = transformer.transform(x_test_counts)

    # Load the SVC model
    svc = LinearSVC()
    svc.fit(x_train_tfidf, y_train)
    y_pred2 = svc.predict(x_test_tfidf)
    # print("Accuracy: "+str(accuracy_score(y_test, y_pred2)))

    # Load Logistic Regression
    lr = LogisticRegression(C=3, max_iter=1000, n_jobs=-1)
    lr.fit(x_train_tfidf, y_train)
    y_pred1 = lr.predict(x_test_tfidf)
    #print("Accuracy: "+str(accuracy_score(y_test,y_pred1)))

    # Load Random Forest Classifier
    rfc = RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=42, class_weight='balanced')
    rfc.fit(x_train_tfidf, y_train)
    y_pred4 = rfc.predict(x_test_tfidf)
    #print("Accuracy: "+str(accuracy_score(y_test,y_pred4)))

    # Load multinomial naive bayes
    mnb = MultinomialNB()
    mnb.fit(x_train_tfidf, y_train)
    y_pred3 = mnb.predict(x_test_tfidf)
    # print("Accuracy: "+str(accuracy_score(y_test,y_pred3)))
    # print(classification_report(y_test, y_pred3))

    # Ensemble Learning
    mnb = MultinomialNB()
    rfc = RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=42)
    lr = LogisticRegression(C=3, max_iter=1000, n_jobs=-1)
    svc = LinearSVC()
    ec = VotingClassifier(estimators=[('Multinominal NB', mnb), ('Random Forest', rfc), (
        'Logistic Regression', lr), ('Support Vector Machine', svc)], voting='hard', weights=[1, 2, 3, 4])
    ec.fit(x_train_tfidf, y_train)
    print(accuracy_score(y_test, y_pred6))

    # Save the model
    l = []
    l.append(ec)
    l.append(count_vect)
    l.append(transformer)
    joblib.dump(l, 'Text_SVM.pkl')


classifyModel()
# source = '''I. Acceptance of terms
# Thank you for using Zomato. These Terms of Service (the "Terms") are intended to make you aware of your legal rights and responsibilities with respect to your access to and use of the Zomato website at www.zomato.com (the "Site") and any related mobile or software applications ("Zomato Platform") including but not limited to delivery of information via the website whether existing now or in the future that link to the Terms (collectively, the "Services").

# These Terms are effective for all existing and future Zomato customers, including but without limitation to users having access to 'restaurant business page' to manage their claimed business listings.

# Please read these Terms carefully. By accessing or using the Zomato Platform, you are agreeing to these Terms and concluding a legally binding contract with Zomato Limited (formerly known as Zomato Private Limited and Zomato Media Private Limited) and/or its affiliates (excluding Zomato Foods Private Limited) (hereinafter collectively referred to as "Zomato"). You may not use the Services if you do not accept the Terms or are unable to be bound by the Terms. Your use of the Zomato Platform is at your own risk, including the risk that you might be exposed to content that is objectionable, or otherwise inappropriate.

# In order to use the Services, you must first agree to the Terms. You can accept the Terms by:

# Clicking to accept or agree to the Terms, where it is made available to you by Zomato in the user interface for any particular Service; or
# Actually using the Services. In this case, you understand and agree that Zomato will treat your use of the Services as acceptance of the Terms from that point onwards.
# II. Definitions
# Customer
# "Customer" or "You" or "Your" refers to you, as a customer of the Services. A customer is someone who accesses or uses the Services for the purpose of sharing, displaying, hosting, publishing, transacting, or uploading information or views or pictures and includes other persons jointly participating in using the Services including without limitation a user having access to 'restaurant business page' to manage claimed business listings or otherwise.

# Content
# "Content" will include (but is not limited to) reviews, images, photos, audio, video, location data, nearby places, and all other forms of information or data. "Your content" or "Customer Content" means content that you upload, share or transmit to, through or in connection with the Services, such as likes, ratings, reviews, images, photos, messages, chat communication, profile information, or any other materials that you publicly display or displayed in your account profile. "Zomato Content" means content that Zomato creates and make available in connection with the Services including, but not limited to, visual interfaces, interactive features, graphics, design, compilation, computer code, products, software, aggregate ratings, reports and other usage-related data in connection with activities associated with your account and all other elements and components of the Services excluding Your Content and Third Party Content. "Third Party Content" means content that comes from parties other than Zomato or its Customers, such as Restaurant Partners and is available on the Services.

# Restaurant(s)
# "Restaurant" means the restaurants listed on Zomato Platform.

# III. Eligibility to use the services
# You hereby represent and warrant that you are at least eighteen (18) years of age or above and are fully able and competent to understand and agree the terms, conditions, obligations, affirmations, representations, and warranties set forth in these Terms.

# Compliance with Laws. You are in compliance with all laws and regulations in the country in which you live when you access and use the Services. You agree to use the Services only in compliance with these Terms and applicable law, and in a manner that does not violate our legal rights or those of any third party(ies).

# IV. Changes to the terms
# Zomato may vary or amend or change or update these Terms, from time to time entirely at its own discretion. You shall be responsible for checking these Terms from time to time and ensure continued compliance with these Terms. Your use of Zomato Platform after any such amendment or change in the Terms shall be deemed as your express acceptance to such amended/changed terms and you also agree to be bound by such changed/amended Terms.

# V. Translation of the terms
# Zomato may provide a translation of the English version of the Terms into other languages. You understand and agree that any translation of the Terms into other languages is only for your convenience and that the English version shall govern the terms of your relationship with Zomato. Furthermore, if there are any inconsistencies between the English version of the Terms and its translated version, the English version of the Terms shall prevail over others.

# VI. Provision of the services being offered by Zomato
# Zomato is constantly evolving in order to provide the best possible experience and information to its Customers. You acknowledge and agree that the form and nature of the Services which Zomato provides, may require affecting certain changes in it, therefore, Zomato reserves the right to suspend/cancel, or discontinue any or all products or services at any time without notice, make modifications and alterations in any or all of its contents, products and services contained on the site without any prior notice.

# We, the software, or the software application store that makes the software available for download may include functionality to automatically check for updates or upgrades to the software. Unless your device, its settings, or computer software does not permit transmission or use of upgrades or updates, you agree that we, or the applicable software or software application store, may provide notice to you of the availability of such upgrades or updates and automatically push such upgrade or update to your device or computer from time-to-time. You may be required to install certain upgrades or updates to the software in order to continue to access or use the Services, or portions thereof (including upgrades or updates designed to correct issues with the Services). Any updates or upgrades provided to you by us under the Terms shall be considered part of the Services.

# You acknowledge and agree that if Zomato disables access to your account, you may be prevented from accessing the Services, your account details or any files or other content, which is contained in your account.

# You acknowledge and agree that while Zomato may not currently have set a fixed upper limit on the number of transmissions you may send or receive through the Services, Zomato may set such fixed upper limits at any time, at Zomato's discretion.

# In our effort to continuously improve the Zomato Platform and Services, we undertake research and conduct experiments from time to time on various aspects of the Services and offerings, including our apps, websites, user interface and promotional campaigns. As a result of which, some Customers may experience features differently than others at any given time. This is for making the Zomato Platform better, more convenient and easy to use, improving Customer experience, enhancing the safety and security of our services and offerings and developing new services and features.

# By using Zomato's Services you agree to the following disclaimers:

# The Content on these Services is for informational purposes only. Zomato disclaims any liability for any information that may have become outdated since the last time the particular piece of information was updated. Zomato reserves the right to make changes and corrections to any part of the Content on these Services at any time without prior notice. Zomato does not guarantee the quality of the Goods, the prices listed in menus or the availability of all menu items at any Restaurant/Merchant. Unless stated otherwise, all pictures and information contained on these Services are believed to be owned by or licensed to Zomato. Please email a takedown request (by using the "Contact Us" link on the home page) to the webmaster if you are the copyright owner of any Content on these Services and you think the use of the above material violates Your copyright in any way. Please indicate the exact URL of the webpage in your request. All images shown here have been digitized by Zomato. No other party is authorized to reproduce or republish these digital versions in any format whatsoever without the prior written permission of Zomato.
# Any certification, licenses or permits ("Certification") or information in regard to such Certification that may be displayed on the Restaurant's listing page on the Zomato Platform is for informational purposes only. Such Certification is displayed by Zomato on an 'as available' basis that is provided to Zomato by the Restaurant partner(s)/Merchant(s). Zomato does not make any warranties about the validity, authenticity, reliability and accuracy of such Certification or any information displayed in this regard. Any reliance by a Customer upon the Certification or information thereto shall be strictly at such Customer's own risk and Zomato in no manner shall assume any liability whatsoever for any losses or damages in connection with the use of this information or for any inaccuracy, invalidity or discrepancy in the Certification or non-compliance of any applicable local laws or regulations by the Restaurant partner/Merchant.
# Zomato reserves the right to charge a subscription and/or membership fee in respect of any of its product or service and/or any other charge or fee on a per order level from Customers, in respect of any of its product or service on the Zomato Platform anytime in future.

# Zomato may from time to time introduce referral and/or incentive based programs for its Customers (Program). These Program(s) may be governed by their respective terms and conditions. By participating in the Program, Customers are bound by the Program terms and conditions as well as the Zomato Platform terms. Further, Zomato reserves the right to terminate / suspend the Customer's account and/or credits / points earned and/or participation of the Customer in the Program if Zomato determines in its sole discretion that the Customer has violated the rules of the Program and/or has been involved in activities that are in contravention of the Program terms and/or Zomato Platform terms or has engaged in activities which are fraudulent / unlawful in nature. Furthermore, Zomato reserves the right to modify, cancel and discontinue its Program without notice to the Customer.

# Zomato may from time to time offer to the Customers credits, promo codes, vouchers or any other form of cashback that Zomato may decide at its discretion. Zomato reserves the right to modify, convert, cancel and/or discontinue such credits, promo codes or vouchers, as it may deem fit.

# VII. Use of services by you or Customer
# 1. Zomato Customer Account Including 'Claim Your Business Listing' Access
# a. You must create an account in order to use some of the features offered by the Services, including without limitation to 'claim your business listing' on the Services. Use of any personal information you provide to us during the account creation process is governed by our Privacy Policy. You must keep your password confidential and you are solely responsible for maintaining the confidentiality and security of your account, all changes and updates submitted through your account, and all activities that occur in connection with your account.

# b. You may also be able to register to use the Services by logging into your account with your credentials from certain third party social networking sites (e.g., Facebook). You confirm that you are the owner of any such social media account and that you are entitled to disclose your social media login information to us. You authorize us to collect your authentication information, and other information that may be available on or through your social media account consistent with your applicable settings and instructions.

# c. In creating an account and/or claiming your business' listing, you represent to us that all information provided to us in such process is true, accurate and correct, and that you will update your information as and when necessary in order to keep it accurate. If you are creating an account or claiming a business listing, then you represent to us that you are the owner or authorized agent of such business. You may not impersonate someone else, create or use an account for anyone other than yourself, provide an email address other than your own, create multiple accounts or business listings except as otherwise authorized by us, or provide or use false information to obtain access to a business' listing on the Services that you are not legally entitled to claim. You acknowledge that any false claiming of a business listing may cause Zomato or third parties to incur substantial economic damages and losses for which you may be held liable and accountable.

# d. You are also responsible for all activities that occur in your account. You agree to notify us immediately of any unauthorized use of your account in order to enable us to take necessary corrective action. You also agree that you will not allow any third party to use your Zomato account for any purpose and that you will be liable for such unauthorized access.

# e. By creating an account, you agree to receive certain communications in connection with Zomato Platform or Services. For example, you might receive comments from other Customers or other Customers may follow the activity to do on your account. You can opt-out or manage your preferences regarding non-essential communications through account settings.

# 2. Others Terms
# a. In order to connect you to certain restaurants, we provide value added telephony services through our phone lines, which are displayed on the specific restaurant listing page on the Zomato Platform, which connect directly to restaurants' phone lines. We record all information regarding this call including the voice recording of the conversation between you, and the restaurant (for internal billing tracking purposes and customer service improvement at the restaurant's end). If you do not wish that your information be recorded in such a manner, please do not use the telephone services provided by Zomato. You explicitly agree and permit Zomato to record all this information when you avail the telephony services through the Zomato provided phone lines on the Zomato Platform.

# b. You agree to use the Services only for purposes that are permitted by (a) the Terms and (b) any applicable law, regulation or generally accepted practices or guidelines in the relevant jurisdictions.

# c. You agree to use the data owned by Zomato (as available on the Services or through any other means like API etc.) only for personal use/purposes and not for any commercial use (other than in accordance with 'Claim Your Business Listing' access) unless agreed to by/with Zomato in writing.

# d. You agree not to access (or attempt to access) any of the Services by any means other than the interface that is provided by Zomato, unless you have been specifically allowed to do so, by way of a separate agreement with Zomato. You specifically agree not to access (or attempt to access) any of the Services through any automated means (including use of scripts or web crawlers) and shall ensure that you comply with the instructions set out in any robots.txt file present on the Services.

# e. You agree that you will not engage in any activity that interferes with or disrupts the Services (or the servers and networks which are connected to the Services). You shall not delete or revise any material or information posted by any other Customer(s), shall not engage in spamming, including but not limited to any form of emailing, posting or messaging that is unsolicited.

# VIII. Content
# 1. Ownership of Zomato Content and Proprietary Rights
# a. We are the sole and exclusive copyright owners of the Services and our Content. We also exclusively own the copyrights, trademarks, service marks, logos, trade names, trade dress and other intellectual and proprietary rights throughout the world (the "IP Rights") associated with the Services and Zomato Content, which may be protected by copyright, patent, trademark and other applicable intellectual property and proprietary rights and laws. You acknowledge that the Services contain original works and have been developed, compiled, prepared, revised, selected, and arranged by us and others through the application of methods and standards of judgment developed and applied through the expenditure of substantial time, effort, and money and constitutes valuable intellectual property of us and such others. You further acknowledge that the Services may contain information which is designated as confidential by Zomato and that you shall not disclose such information without Zomato's prior written consent.

# b. You agree to protect Zomato's proprietary rights and the proprietary rights of all others having rights in the Services during and after the term of this agreement and to comply with all reasonable written requests made by us or our suppliers and licensors of content or otherwise to protect their and others' contractual, statutory, and common law rights in the Services. You acknowledge and agree that Zomato (or Zomato's licensors) own all legal right, title and interest in and to the Services, including any IP Rights which subsist in the Services (whether those rights happen to be registered or not, and wherever in the world those rights may exist). You further acknowledge that the Services may contain information which is designated as confidential by Zomato and that you shall not disclose such information without Zomato's prior written consent. Unless you have agreed otherwise in writing with Zomato, nothing in the Terms gives you a right to use any of Zomato's trade names, trademarks, service marks, logos, domain names, and other distinctive brand features.

# c. You agree not to use any framing techniques to enclose any trademark or logo or other proprietary information of Zomato; or remove, conceal or obliterate any copyright or other proprietary notice or source identifier, including without limitation, the size, colour, location or style of any proprietary mark(s). Any infringement shall lead to appropriate legal proceedings against you at an appropriate forum for seeking all available/possible remedies under applicable laws of the country of violation. You cannot modify, reproduce, publicly display or exploit in any form or manner whatsoever any of the Zomato's Content in whole or in part except as expressly authorized by Zomato.

# d. To the fullest extent permitted by applicable law, we neither warrant nor represent that your use of materials displayed on the Services will not infringe rights of third parties not owned by or affiliated with us. You agree to immediately notify us upon becoming aware of any claim that the Services infringe upon any copyright trademark, or other contractual, intellectual, statutory, or common law rights by following the instructions contained below in section XVI.

# 2. Your License to Zomato Content
# a. We grant you a personal, limited, non-exclusive and non-transferable license to access and use the Services only as expressly permitted in these Terms. You shall not use the Services for any illegal purpose or in any manner inconsistent with these Terms. You may use information made available through the Services solely for your personal, non-commercial use. You agree not to use, copy, display, distribute, modify, broadcast, translate, reproduce, reformat, incorporate into advertisements and other works, sell, promote, create derivative works, or in any way exploit or allow others to exploit any of Zomato Content in whole or in part except as expressly authorized by us. Except as otherwise expressly granted to you in writing, we do not grant you any other express or implied right or license to the Services, Zomato Content or our IP Rights.

# b. Any violation by you of the license provisions contained in this Section may result in the immediate termination of your right to use the Services, as well as potential liability for copyright and other IP Rights infringement depending on the circumstances.

# 3. Zomato License to Your or Customer Content
# In consideration of availing the Services on the Zomato Platform and by submitting Your Content, you hereby irrevocably grant Zomato a perpetual, irrevocable, world-wide, non-exclusive, fully paid and royalty-free, assignable, sub-licensable and transferable license and right to use Your Content (including content shared by any business user having access to a 'restaurant business page' to manage claimed business listings or otherwise) and all IP Rights therein for any purpose including API partnerships with third parties and in any media existing now or in future. By "use" we mean use, copy, display, distribute, modify, translate, reformat, incorporate into advertisements and other works, analyze, promote, commercialize, create derivative works, and in the case of third party services, allow their users and others to do the same. You grant us the right to use the name or username that you submit in connection with Your Content. You irrevocably waive, and cause to be waived, any claims and assertions of moral rights or attribution with respect to Your Content brought against Zomato or its Customers, any third party services and their users.

# 4. Representations Regarding Your or Customer Content
# a. You are responsible for Your Content. You represent and warrant that you are the sole author of, own, or otherwise control all of the rights of Your Content or have been granted explicit permission from the rights holder to submit Your Content; Your Content was not copied from or based in whole or in part on any other content, work, or website; Your Content was not submitted via the use of any automated process such as a script bot; use of Your Content by us, third party services, and our and any third party users will not violate or infringe any rights of yours or any third party; Your Content is truthful and accurate; and Your Content does not violate the Guidelines and Policies or any applicable laws

# b. If Your Content is a review, you represent and warrant that you are the sole author of that review; the review reflects an actual dining experience that you had; you were not paid or otherwise remunerated in connection with your authoring or posting of the review; and you had no financial, competitive, or other personal incentive to author or post a review that was not a fair expression of your honest opinion.

# c. You assume all risks associated with Your Content, including anyone's reliance on its quality, accuracy, or reliability, or any disclosure by you of information in Your Content that makes you personally identifiable. While we reserve the right to remove Content, we do not control actions or Content posted by our Customers and do not guarantee the accuracy, integrity or quality of any Content. You acknowledge and agree that Content posted by Customers and any and all liability arising from such Content is the sole responsibility of the Customer who posted the content, and not Zomato.

# 5. Content Removal
# We reserve the right, at any time and without prior notice, to remove, block, or disable access to any Content that we, for any reason or no reason, consider to be objectionable, in violation of the Terms or otherwise harmful to the Services or our Customers in our sole discretion. Subject to the requirements of applicable law, we are not obligated to return any of Your Content to you under any circumstances. Further, the Restaurant reserves the right to delete any images and pictures forming part of Customer Content, from such Restaurant's listing page at its sole discretion.

# 6. Third Party Content and Links
# a. Some of the content available through the Services may include or link to materials that belong to third parties, such as third party reservation services or food delivery/ordering or dining out. Please note that your use of such third party services will be governed by the terms of service and privacy policy applicable to the corresponding third party. We may obtain business addresses, phone numbers, and other contact information from third party vendors who obtain their data from public sources.

# b. We have no control over, and make no representation or endorsement regarding the accuracy, relevancy, copyright compliance, legality, completeness, timeliness or quality of any product, services, advertisements and other content appearing in or linked to from the Services. We do not screen or investigate third party material before or after including it on our Services.

# c. We reserve the right, in our sole discretion and without any obligation, to make improvements to, or correct any error or omissions in, any portion of the content accessible on the Services. Where appropriate, we may in our sole discretion and without any obligation, verify any updates, modifications, or changes to any content accessible on the Services, but shall not be liable for any delay or inaccuracies related to such updates. You acknowledge and agree that Zomato is not responsible for the availability of any such external sites or resources, and does not endorse any advertising, products or other materials on or available from such web sites or resources.

# d. Third party content, including content posted by our Customers or Restaurant Partners, does not reflect our views or that of our parent, subsidiary, affiliate companies, branches, employees, officers, directors, or shareholders. In addition, none of the content available through the Services is endorsed or certified by the providers or licensors of such third party content. We assume no responsibility or liability for any of Your Content or any third party content.

# e. You further acknowledge and agree that Zomato is not liable for any loss or damage which may be incurred by you as a result of the availability of those external sites or resources, or as a result of any reliance placed by you on the completeness, accuracy or existence of any advertising, products or other materials on, or available from, such websites or resources. Without limiting the generality of the foregoing, we expressly disclaim any liability for any offensive, defamatory, illegal, invasive, unfair, or infringing content provided by third parties.

# 7. Customer Reviews
# a. Customer reviews or ratings for Restaurants do not reflect the opinion of Zomato. Zomato receives multiple reviews or ratings for Restaurants by Customers, which reflect the opinions of the Customers. It is pertinent to state that each and every review posted on Zomato is the personal opinion of the Customer/reviewer only. Zomato is a neutral platform, which solely provides a means of communication between Customers/reviewers including Customers or restaurant owners/representatives with access to restaurant business page. The advertisements published on the Zomato Platform are independent of the reviews received by such advertisers.

# b. We are a neutral platform and we don't arbitrate disputes, however in case if someone writes a review that the restaurant does not consider to be true, the best option for the restaurant representative would be to contact the reviewer or post a public response in order to clear up any misunderstandings. If the Restaurant believes that any particular Customer's review violates any of the Zomato' policies, the restaurant may write to us at neutrality@zomato.com and bring such violation to our attention. Zomato may remove the review in its sole discretion if review is in violation of the Terms, or content guidelines and policies or otherwise harmful to the Services

# IX. Content guidelines and privacy policy
# 1. Content Guidelines
# You represent that you have read, understood and agreed to our Guidelines and Polices related to Content

# 2. Privacy Policy
# You represent that you have read, understood and agreed to our Privacy Policy. Please note that we may disclose information about you to third parties or government authorities if we believe that such a disclosure is reasonably necessary to (i) take action regarding suspected illegal activities; (ii) enforce or apply our Terms and Privacy Policy; (iii) comply with legal process or other government inquiry, such as a search warrant, subpoena, statute, judicial proceeding, or other legal process/notice served on us; or (iv) protect our rights, reputation, and property, or that of our Customers, affiliates, or the general public

# X. Restrictions on use
# Without limiting the generality of these Terms, in using the Services, you specifically agree not to post or transmit any content (including review) or engage in any activity that, in our sole discretion:

# a. Violate our Guidelines and Policies;

# b. Is harmful, threatening, abusive, harassing, tortious, indecent, defamatory, discriminatory, vulgar, profane, obscene, libellous, hateful or otherwise objectionable, invasive of another's privacy, relating or encouraging money laundering or gambling;

# c. Constitutes an inauthentic or knowingly erroneous review, or does not address the goods and services, atmosphere, or other attributes of the business you are reviewing.

# d. Contains material that violates the standards of good taste or the standards of the Services;

# e. Violates any third-party right, including, but not limited to, right of privacy, right of publicity, copyright, trademark, patent, trade secret, or any other intellectual property or proprietary rights;

# f. Accuses others of illegal activity, or describes physical confrontations;

# g. Alleges any matter related to health code violations requiring healthcare department reporting. Refer to our Guidelines and Policies for more details about health code violations.

# h. Is illegal, or violates any federal, state, or local law or regulation (for example, by disclosing or trading on inside information in violation of securities law);

# i. Attempts to impersonate another person or entity;

# j. Disguises or attempts to disguise the origin of Your Content, including but not limited to by: (i) submitting Your Content under a false name or false pretences; or (ii) disguising or attempting to disguise the IP address from which Your Content is submitted;

# k. Constitutes a form of deceptive advertisement or causes, or is a result of, a conflict of interest;

# l. Is commercial in nature, including but not limited to spam, surveys, contests, pyramid schemes, postings or reviews submitted or removed in exchange for payment, postings or reviews submitted or removed by or at the request of the business being reviewed, or other advertising materials;

# m. Asserts or implies that Your Content is in any way sponsored or endorsed by us;

# n. Contains material that is not in English or, in the case of products or services provided in foreign languages, the language relevant to such products or services;

# o. Falsely states, misrepresents, or conceals your affiliation with another person or entity;

# p. Accesses or uses the account of another customer without permission;

# q. Distributes computer viruses or other code, files, or programs that interrupt, destroy, or limit the functionality of any computer software or hardware or electronic communications equipment;

# r. Interferes with, disrupts, or destroys the functionality or use of any features of the Services or the servers or networks connected to the Services;

# s. "Hacks" or accesses without permission our proprietary or confidential records, records of another Customer, or those of anyone else;

# t. Violates any contract or fiduciary relationship (for example, by disclosing proprietary or confidential information of your employer or client in breach of any employment, consulting, or non-disclosure agreement);

# u. Decompiles, reverse engineers, disassembles or otherwise attempts to derive source code from the Services;

# v. Removes, circumvents, disables, damages or otherwise interferes with security-related features, or features that enforce limitations on use of, the Services;

# w. Violates the restrictions in any robot exclusion headers on the Services, if any, or bypasses or circumvents other measures employed to prevent or limit access to the Services;

# x. Collects, accesses, or stores personal information about other Customers of the Services;

# y. Is posted by a bot;

# z. Harms minors in any way;

# aa. Threatens the unity, integrity, defense, security or sovereignty of India or of the country of use, friendly relations with foreign states, or public order or causes incitement to the commission of any cognizable offence or prevents investigation of any offence or is insulting any other nation;

# ab. Modifies, copies, scrapes or crawls, displays, publishes, licenses, sells, rents, leases, lends, transfers or otherwise commercialize any rights to the Services or Our Content; or

# ac. Attempts to do any of the foregoing.

# ad. is patently false and untrue, and is written or published in any form, with the intent to mislead or harass a person, entity or agency for financial gain or to cause any injury to any person;

# You acknowledge that Zomato has no obligation to monitor your – or anyone else's – access to or use of the Services for violations of the Terms, or to review or edit any content. However, we have the right to do so for the purpose of operating and improving the Services (including without limitation for fraud prevention, risk assessment, investigation and customer support purposes), to ensure your compliance with the Terms and to comply with applicable law or the order or requirement of legal process, a court, consent decree, administrative agency or other governmental body

# You hereby agree and assure Zomato that the Zomato Platform/Services shall be used for lawful purposes only and that you will not violate laws, regulations, ordinances or other such requirements of any applicable Central, Federal State or local government or international law(s). You shall not upload, post, email, transmit or otherwise make available any unsolicited or unauthorized advertising, promotional materials, junk mail, spam mail, chain letters or any other form of solicitation, encumber or suffer to exist any lien or security interest on the subject matter of these Terms or to make any representation or warranty on behalf of Zomato in any form or manner whatsoever.

# You hereby agree and assure that while communicating on the Zomato Platform including but not limited to giving cooking instructions to the Restaurants, communicating with our support agents on chat support or with the Delivery Partners, through any medium, You shall not use abusive and derogatory language and/or post any objectionable information that is unlawful, threatening, defamatory, or obscene. In the event you use abusive language and/or post objectionable information, Zomato reserves the right to suspend the chat support service and/or block your access and usage of the Zomato Platform, at any time with or without any notice.

# Any Content uploaded by you, shall be subject to relevant laws of India and of the country of use and may be disabled, or and may be subject to investigation under applicable laws. Further, if you are found to be in non-compliance with the laws and regulations, these terms, or the privacy policy of the Zomato Platform, Zomato shall have the right to immediately block your access and usage of the Zomato Platform and Zomato shall have the right to remove any non-compliant content and or comment forthwith, uploaded by you and shall further have the right to take appropriate recourse to such remedies as would be available to it under various statutes.

# XI. Customer feedback
# If you share or send any ideas, suggestions, changes or documents regarding Zomato's existing business ("Feedback"), you agree that (i) your Feedback does not contain the confidential, secretive or proprietary information of third parties, (ii) Zomato is under no obligation of confidentiality with respect to such Feedback, and shall be free to use the Feedback on an unrestricted basis (iii) Zomato may have already received similar Feedback from some other Customer or it may be under consideration or in development, and (iv) By providing the Feedback, you grant us a binding, non-exclusive, royalty-free, perpetual, global license to use, modify, develop, publish, distribute and sublicense the Feedback, and you irrevocably waive, against Zomato and its Customers any claims/assertions, whatsoever of any nature, with regard to such Feedback.
# Please provide only specific Feedback on Zomato's existing products or marketing strategies; do not include any ideas that Zomato's policy will not permit it to accept or consider.
# Notwithstanding the abovementioned clause, Zomato or any of its employees do not accept or consider unsolicited ideas, including ideas for new advertising campaigns, new promotions, new or improved products or technologies, product enhancements, processes, materials, marketing plans or new product names. Please do not submit any unsolicited ideas, original creative artwork, suggestions or other works ("Submissions") in any form to Zomato or any of its employees.
# The purpose of this policy is to avoid potential misunderstandings or disputes when Zomato's products or marketing strategies might seem similar to ideas submitted to Zomato. If, despite our request to not send us your ideas, you still submit them, then regardless of what your letter says, the following terms shall apply to your Submissions.
# Terms of Idea Submission
# You agree that: (1) your Submissions and their Contents will automatically become the property of Zomato, without any compensation to you; (2) Zomato may use or redistribute the Submissions and their contents for any purpose and in any way; (3) there is no obligation for Zomato to review the Submission; and (4) there is no obligation to keep any Submissions confidential.
# XII. Advertising
# Some of the Services are supported by advertising revenue and may display advertisements and promotions. These advertisements may be targeted to the content of information stored on the Services, queries made through the Services or other information. The manner, mode and extent of advertising by Zomato on the Services are subject to change without specific notice to you. In consideration for Zomato granting you access to and use of the Services, you agree that Zomato may place such advertising on the Services.
# Part of the site may contain advertising information or promotional material or other material submitted to Zomato by third parties or Customers. Responsibility for ensuring that material submitted for inclusion on the Zomato Platform or mobile apps complies with applicable international and national law is exclusively on the party providing the information/material. Your correspondence or business dealings with, or participation in promotions of, advertisers other than Zomato found on or through the Zomato Platform and or mobile apps, including payment and delivery of related goods or services, and any other terms, conditions, warranties or representations associated with such dealings, shall be solely between you and such advertiser. Zomato will not be responsible or liable for any error or omission, inaccuracy in advertising material or any loss or damage of any sort incurred as a result of any such dealings or as a result of the presence of such other advertiser(s) on the Zomato Platform and mobile application.
# For any information related to a charitable campaign ("Charitable Campaign") sent to Customers and/or displayed on the Zomato Platform where Customers have an option to donate money by way of (a) payment on a third party website; or (b) depositing funds to a third party bank account, Zomato is not involved in any manner in the collection or utilization of funds collected pursuant to the Charitable Campaign. Zomato does not accept any responsibility or liability for the accuracy, completeness, legality or reliability of any information related to the Charitable Campaign. Information related to the Charitable Campaign is displayed for informational purposes only and Customers are advised to do an independent verification before taking any action in this regard.
# XIII. Additional Terms and Conditions for Customers using the various services offered by Zomato:
# 1. FOOD ORDERING AND DELIVERY:
# a. Zomato provides food ordering and delivery services by entering into contractual arrangements with restaurant partners (“Restaurant Partners”) and Stores (as defined below) on a principal-to-principal basis for the purpose of listing their menu items or the Products (as defined below) for food ordering and delivery by the Customers on the Zomato Platform.

# b. The Customers can access the menu items or Products listed on the Zomato Platform and place orders against the Restaurant Partner(s)/Store(s) through Zomato.

# c. Your request to order food and beverages or Products from a Restaurant Partner or a Store page on the Zomato Platform shall constitute an unconditional and irrevocable authorization issued in favour of Zomato to place orders for food and beverages or Products against the Restaurant Partner(s)/Store(s) on your behalf.

# d. Delivery of an order placed by you through the Zomato Platform may either be undertaken directly by the Restaurant Partner or the Store against whom you have placed an order, or facilitated by Zomato through a third-party who may be available to provide delivery services to you (“Delivery Partners”). In both these cases, Zomato is merely acting as an intermediary between you and the Delivery Partners, or you and the Restaurant Partner or the Store, as the case may be.

# e. The acceptance by a Delivery Partner of undertaking delivery of your order shall constitute a contract of service under the Consumer Protection Act, 2019 or any successor legislations, between you and the Delivery Partner, to which Zomato is not a party under any applicable law. It is clarified that Zomato does not provide any delivery or logistics services and only enables the delivery of food and beverages or Products ordered by the Customers through the Zomato Platform by connecting the Customers with the Delivery Partners or the Restaurant Partners or the Store, as the case may be.

# f. Where Zomato is facilitating delivery of an order placed by you, Zomato shall not be liable for any acts or omissions on part of the Delivery Partner including deficiency in service, wrong delivery of order, time taken to deliver the order, order package tampering, etc.

# g. You may be charged a delivery fee for delivery of your order by the Delivery Partner or the Restaurant Partner or the Store, as the Delivery Partner or the Restaurant Partner or the Store may determine (“Delivery Charges"). You agree that Zomato is authorized to collect, on behalf of the Restaurant Partner or the Delivery Partner or the Store, the Delivery Charges for the delivery service provided by the Restaurant Partner or the Store or the Delivery Partner, as the case may be. The Delivery Charges may vary from order to order, which may be determined on multiple factors which shall include but not be limited to Restaurant Partner / Store, order value, distance, time of the day. Zomato will inform you of the Delivery Charges that may apply to you, provided you will be responsible for Delivery Charges incurred for your order regardless of your awareness of such Delivery Charges.

# h. In addition to the Delivery Charges, you may also be charged an amount towards delivery surge for delivery of your order facilitated by the Delivery Partner or the Restaurant Partner or the Store, which is determined on the basis of various factors including but not limited to distance covered, time taken, demand for delivery, real time analysis of traffic and weather conditions, seasonal peaks or such other parameters as may be determined from time to time (“Delivery Surge"). You agree that Zomato is authorized to collect, on behalf of the Restaurant Partner or the Delivery Partner or the Store, the Delivery Surge for the delivery service provided by the Restaurant Partner or the Store or the Delivery Partner, as the case may be. The Delivery Surge may vary from order to order, which may be determined on multiple factors which shall include but not be limited to Restaurant Partner / Store, order value, distance, demand during peak hours. Zomato will use reasonable efforts to inform you of the Delivery Surge that may apply to you, provided you will be responsible for the Delivery Surge incurred for your order regardless of your awareness of such Delivery Surge.

# i. In respect of the order placed by You, Zomato shall issue documents like order summary, tax invoices, etc. as per the applicable legal regulations and common business practices.

# j. You are expected to respect the dignity and diversity of Delivery Partners and accordingly you agree to not discriminate against any Delivery Partner on the basis of Discrimination Characteristics (as defined below). You are also expected to enable provision of a secure and fearless gig/ platform work environment for the delivery partners including prevention and deterrence of harassment (including sexual harassment) towards Delivery Partners.

# Discrimination Characteristics shall mean discrimination based on race, community, religion, disability, gender, sexual orientation, gender identity, age (insofar as permitted by applicable laws to undertake the relevant gig work), genetic information, or any other legally protected status.

# A. Food Ordering and Delivery with Restaurant Partners:
# a. All prices listed on the Zomato Platform are provided by the Restaurant Partner, including packaging or handling charges, if any, at the time of publication on the Zomato Platform and the same are displayed by Zomato as received from the Restaurant Partner. While we take great care to keep them up to date, the final price charged to you by the Restaurant Partner, including the packaging and handling charges may change at the time of delivery. In the event of a conflict between price on the Zomato Platform and price charged by the Restaurant Partner, the price charged by the Restaurant Partner shall be deemed to be the correct price except Delivery Charge of Zomato.

# b. On Time Guarantee: Orders placed at select Restaurant Partners may have this service available. When enabled, Zomato uses its technology platform to allocate a suitable delivery partner, who provides the service, in such a way that it minimises the delays in the orders. This includes prioritising allocation of Delivery Partners, along with making sure these orders are not clubbed with any other orders. However, you acknowledge that such services are facilitated by the Delivery Partner on a best effort basis, hence should your order fail to reach you on or prior to the On Time Guarantee Time, you would be eligible to claim and receive a coupon of up to 100% of your order value. You will be required to claim the Coupon within twenty four (24) hours from the time such Order is delivered to you failing which your eligibility to receive the Coupon will expire. Further the validity period of the Coupon would be 3 (three) days from receipt thereof. Notwithstanding anything set out herein above, you shall not be eligible to receive the Coupon if:

# i. Delay on the On Time Guarantee Time is for unforeseen reasons eg. strikes, natural disaster, Restaurant Partner’s inability to provide the Order.
# ii. You change, edit, modify or cancel such Order or do any such act that has the effect of changing, editing or modifying such order including but not limited to adding or changing the items ordered, receiving delivery at a location other than the one indicated at the time of placing of the order etc.
# iii. You indulge in actions intended to delay the order including but not limited to asking the Delivery Partner to delay the Order, becoming unresponsive on call etc.
# iv. The order is a bulk order (as per Zomato’s standard order size)
# v. The order is cancelled due to any reason.

# For the purpose of this clause, words capitalised shall have the following meaning: “On Time Guarantee Time" shall mean the Promised Delivery Time, which starts when the order is accepted by the Restaurant Partner. The actual delivery time will be counted as the period between the Restaurant Partner accepting the order and the Delivery Partner reaching within 100 metre radius from your location or first barrier point (security guard/reception etc.) whichever is earlier. “Coupon" shall mean one-time code generated by Zomato for delay in On Time Guarantee Time to be used as per applicable conditions.

# c. Zomato Gold For Food Ordering and Delivery: Zomato Gold members in India can avail Discounts (as defined below) extended by Partner Restaurants (as defined below) on home delivery. Please refer to the terms and conditions set out below in clause 4.

# d. Zomato Gift

# i. You may place an order with a Restaurant Partner to be delivered to someone else, your loved ones (may or may not be a Zomato customer) (“Gift Recipient”) as a gift (“Gift Order”). ii. To place a Gift Order, You will be required to provide the Gift Recipient’s contact details, such as name, phone number, address or any other information that may be reasonably required (“Contact Information”) to enable the Restaurant Partner, Delivery Partner deliver the Gift Order.

# iii. By availing Zomato Gift feature, You warrant and represent that You have obtained the Gift Recipient’s consent to provide Zomato with the Contact Information. You hereby further warrant and represent to indemnify and hold Zomato, its directors, employees, affiliates and subsidiaries and their respective directors, employees, harmless against any claims or disputes initiated by the Gift Recipient whose Contact Information was provided by You for the purpose of placing the Gift Order.

# iv. By placing a Gift Order, You hereby irrevocably undertake to be responsible for any refusal made by the Gift Recipient or for any prejudice suffered by the latter or Zomato.

# v. In case the Gift Recipient is, non contactable, the Restaurant Partner, Delivery Partner or we may contact You for further assistance.

# vi. You may send a message with the Gift order which we will endeavour to deliver, however sometimes, the message can’t be sent.

# vii. You will not be charged any additional payment for Gift Order. All charges will be in a similar manner as for a regular placed online order.

# viii. In the event, the Gift Recipient wishes to raise any issue with regard to the Gift Order, they can do so by requesting you to do it or via the chat support on the Zomato Platform or write to us at order@zomato.com.

# ix. Any refund on the Gift order shall be provided to the sender and the Gift Recipient shall not receive any benefit.

# x. You explicitly and unambiguously consent to the collection, use and transfer, in electronic or other forms, of personal information for the purposes of Zomato Gift. You will be required to share certain personal information with Zomato including your name, phone number, email address, payment details and Zomato will use these details in accordance with the Privacy Policy published on www.zomato.com/privacy.

# xi. All other terms and conditions for food ordering and delivery services provided herein under clause XIII[1] shall apply as is.

# B. General Terms and Conditions
# a. Zomato is not a manufacturer, seller or distributor of food and beverages or Products and merely places an order against the Restaurant Partner(s)/Store(s) on behalf of the Customers pursuant to the unconditional and irrevocable authority granted by the Customers to Zomato, and facilitates the sale and purchase of food and beverages or Products between Customers and Restaurant Partners/Store(s), under the contract for sale and purchase of food and beverages or Products between the Customers and Restaurant Partners/Store(s).

# b. Zomato shall not be liable for any acts or omissions on part of the Restaurant Partner/Store(s) including deficiency in service, wrong delivery of order / order mismatch, quality, incorrect pricing, deficient quantity, time taken to prepare or deliver the order, etc.

# c. The Restaurant Partner(s)/Store(s) shall be solely responsible for any warranty/guarantee of the food and beverages or Products sold to the Customer and in no event shall be the responsibility of Zomato.

# d. For the Customers in India, it is hereby clarified by Zomato that the liability of any violation of the applicable rules and regulations made thereunder shall solely rest with the sellers/brand owners, vendors, Restaurant Partner(s)/Store(s), importers or manufacturers of the food products, Products or any Pre Packed Goods accordingly. For the purpose of clarity Pre-Packed Goods shall mean the food and beverages items which is placed in a package of any nature, in such a manner that the contents cannot be changed without tampering it and which is ready for sale to the customer or as may be defined under the Food Safety and Standards Act, 2006 from time to time.

# e. Please note that some of the food and beverages or Products may be suitable for certain ages only. You should check the dish you are ordering and read its description, if provided, prior to placing your order. Zomato shall not be liable in the event the food and beverages or the Product ordered by You does not meet your dietary or any other requirements and/or restrictions.

# f. While placing an order you shall be required to provide certain details, including without limitation, contact number and delivery address. You agree to take particular care when providing these details and warrant that these details are accurate and complete at the time of placing an Order. By providing these details, you express your acceptance to Zomato's terms and privacy policies.

# g. You or any person instructed by you shall not resell food and beverages or Products purchased via the Zomato Platform.

# h. The total price for food ordered, including the Delivery Charges and other charges, will be displayed on the Zomato Platform when you place your order, which may be rounded up to the nearest amount. Customers shall make full payment towards such food or Products ordered via the Zomato Platform.

# i. Any amount that may be charged to you by Zomato over and above the order value, shall be inclusive of applicable taxes.

# j. Delivery periods/Takeaway time quoted at the time of ordering are approximate only and may vary.

# k. Personal Promo code can only be used by You subject to such terms and conditions set forth by Zomato from time to time.

# l. Cancellation and refund policy:

# i. You acknowledge that (1) your cancellation, or attempted or purported cancellation of an order or (2) cancellation due to reasons not attributable to Zomato, that is, in the event you provide incorrect particulars, contact number, delivery address etc., or that you were unresponsive, not reachable or unavailable for fulfillment of the services offered to you, shall amount to breach of your unconditional and irrevocable authorization in favour of Zomato to place that order against the Restaurant Partners/Store(s) on your behalf (“Authorization Breach"). In the event you commit an Authorization Breach, you shall be liable to pay the liquidated damages of an amount equivalent to the order value. You hereby authorize Zomato to deduct or collect the amount payable as liquidated damages through such means as Zomato may determine in its discretion, including without limitation, by deducting such amount from any payment made towards your next Order

# ii. There may be cases where Zomato is either unable to accept your order or cancels the order, due to reasons including without limitation, technical errors, unavailability of the item(s) ordered, or any other reason attributable to Zomato, Restaurant Partner/Store or Delivery Partner. In such cases, Zomato shall not charge a cancellation charge from you. If the order is cancelled after payment has been charged and you are eligible for a refund of the order value or any part thereof, the said amount will be reversed to you.

# iii. No replacement / refund / or any other resolution will be provided without Restaurant Partner’s/Store(s)’ permission.

# iv. Any complaint, with respect to the order which shall include instances but not be limited to food spillage, foreign objects in food, delivery of the wrong order or food and beverages or Products, poor quality, You will be required to share the proof of the same before any resolution can be provided.

# v. You shall not be entitled to a refund in case instructions placed along with the order are not followed in the form and manner You had intended. Instructions are followed by the Restaurant Partner /Store on a best-efforts basis.

# vi. All refunds shall be processed in the same manner as they are received, unless refunds have been provided to You in the form of credits, refund amount will reflect in your account based on respective banks policies.

# 2. ZOMATO PAY:
# For Zomato Pay users in India:

# In the event a Customer makes a payment for the Bill Amount (as defined below) using Zomato Pay on the Zomato Platform, in the city(ies) in which Zomato Pay is available, following terms and conditions shall be specifically applicable to the Customers:

# a. The Customer can make a payment for the Bill Amount on the Zomato Platform by using any payment method available on the 'Payment' section on the Zomato Platform.

# For the purposes of Zomato Pay, “Bill Amount” shall mean the total amount (including applicable taxes, service charge and other charges, as may be applicable, excluding tip) set out in the dining bill for food and beverages availed by a Customer at the restaurant partnered with Zomato for Zomato Pay.

# b. The Customer acknowledges that upon fulfillment of payment of the Bill Amount via the Zomato Platform, the Customer will be required to show the payment confirmation to the restaurant partnered with Zomato for Zomato Pay.

# c. Upon making a payment for the Bill Amount using Zomato Pay via Zomato Platform, the Customer will be entitled to Zomato Pay Benefits (as defined below), subject to successful payment being made by the Customer.

# For the purposes of Zomato Pay, “Zomato Pay Benefits” shall include but not be limited to either of the following: a. instant discount(s) applicable on the Bill Amount b. additional banking partner offer(s) applicable on the final payable amount, net of other discounts (excluding tips), provided that the final payable amount meets the minimum and maximum order value criteria for a particular offer, if any, as may be decided by the bank from time to time.
# c. scratch card offer(s) with cashback in the form of Zomato Credits, provided the final payable amount is minimum INR 50.

# d. The Customers can make a payment for the Bill Amount using Zomato Pay by either scanning the Zomato QR code at the restaurant partnered with Zomato for Zomato Pay or by searching for the restaurant partnered with Zomato for Zomato Pay on the Zomato Platform and selecting Zomato Pay as the payment method. e. The Customer will be solely responsible to pay the restaurant partnered with Zomato for Zomato Pay, the Bill Amount along with all costs and charges payable for all the other items for which you have placed an Order and which are not covered under the Bill Amount. In the event of a concern raised regarding the payment using Zomato Pay, we shall use our best endeavours to assist you however such payment will be subject to verification and confirmation from the restaurant partnered with Zomato for Zomato Pay.

# f. The Customer acknowledges that in order to receive Zomato Credits, the Customer shall scratch the scratch card offer. Once the scratch card offer is revealed to the Customer, the Customer shall avail the Zomato Credits before the date of expiry of such credits. For the purpose of clarity, the date of expiry of Zomato Credits will be available on each scratch card offer.

# g. Your access to Zomato Pay Benefits shall be subject to receipt of successful payments by Zomato.

# h. The Customer is required to be present at the partner restaurant when using Zomato Pay.

# i. The Customer acknowledges that the Zomato Pay Benefits cannot be clubbed with any ongoing offers by the partner restaurant at the restaurant premise or on items which are being sold at maximum retail price (MRP).

# j. The Customer cannot use Zomato Pay for dine in if the Customer is employed at the same partner restaurant.

# k. Zomato reserves the right to terminate / suspend Zomato Pay Benefits to the Customer, if Zomato determines in its sole discretion that: (i) the Customer has violated the terms of Zomato Pay set out herein, (ii) have been involved in activities that are in contravention of the Zomato Pay terms and/or any terms for the usage of Zomato Platform; or (iii) have engaged in activities which are fraudulent / unlawful in nature while availing any of the services of Zomato.

# l. Zomato reserves the right to block and/or terminate/suspend Zomato Pay Benefits on account of breach of these terms including any fraudulent and suspicious activity while using Zomato Pay.

# m. Zomato Pay and the associated Zomato Pay Benefits will be applicable to the Customers on all days. Provided however, the benefit of instant discount(s) applicable on the Bill Value shall not be applicable for the Customer on Exclusion Days (as set out below).

# n. The instant discounts under Zomato Pay Benefits shall be as follows: a. If a Zomato Gold member uses Zomato Pay via Zomato Platform at a restaurant partnered with Zomato Gold, the discount will be reflected to the Customer as 'Zomato Gold discount' (the Zomato Gold Discount will be applicable along with the Zomato Pay Benefits on the Bill Amount); b. If a Zomato Gold member uses Zomato Pay via Zomato Platform at a restaurant not partnered with Zomato Gold, the discount will be reflected as 'instant discount'; c. If a non-Zomato Gold Customer uses Zomato Pay via Zomato Platform at a restaurant partnered with Zomato Gold, the discount will be reflected as 'instant discount', only if applicable; d. If a non-Zomato Gold Customer uses Zomato Pay via Zomato Platform at a restaurant not partnered with Zomato Gold, the discount will be reflected as 'instant discount'.

# o. The Customer acknowledges that the Edition Wallet and Zomato Pay are two different payment methods available on the Zomato Platform and the Customer shall be entitled to Zomato Pay Benefits only if the payment for the Bill Amount is made using Zomato Pay via the Zomato Platform.

# p. The Customer acknowledges that Zomato Pay is being made available purely on a ‘best effort’ basis and availing Zomato Pay is voluntary.

# q. Zomato reserves the right to modify the Zomato Pay Benefits and/or these Zomato Pay Terms from time to time or at any time, modify or discontinue, temporarily or permanently, Zomato Pay Benefits and/or these Zomato Pay Terms, with or without prior notice and the decision of Zomato shall be final and binding in this regard.

# r. These Zomato Pay Terms do not alter in any way the terms or conditions of any other program or arrangement the Customer may have with Zomato. Termination of Zomato Pay and these Zomato Pay Terms shall have no effect on the Terms of Service governing the contractual relationship between the Customers and Zomato.

# s. For any help or queries, you may reach out to us via chat support or write to us at dining@zomato.com.

# Exclusion Days for the purposes of Zomato Pay

# No.	DATE	EVENT
# 1	December 31 and January 1	New Year's Eve/Day
# 2	As per lunar calendar	Pongal (applicable in Tamil Nadu only)
# 3	February 14	Valentine's Day
# 4	As per lunar calendar	Durga Puja (applicable in West Bengal only)
# 4	As per lunar calendar	Diwali
# 4	December 24 and December 25	Christmas Eve/Day
# For Zomato Pay users in UAE:

# In the event a Customer makes a payment for the Bill Amount (as defined below) using Zomato Pay on the Zomato Platform, in the city(ies) in which Zomato Pay is available, following terms and conditions shall be specifically applicable to the Customers:

# a. The Customer can make a payment for the Bill Amount on the Zomato Platform by using any payment method available on the 'Payment' section on the Zomato Platform. For the purposes of Zomato Pay, “Bill Amount” shall mean the total amount (including applicable taxes, service charge and other charges, as may be applicable, excluding tip) set out in the dining bill for food and beverages availed by a Customer at the restaurant partnered with Zomato for Zomato Pay.

# b. The Customer acknowledges that upon fulfillment of payment of the Bill Amount via the Zomato Platform, the Customer will be required to show the payment confirmation to the restaurant partnered with Zomato for Zomato Pay.

# c. Upon making a payment for the Bill Amount using Zomato Pay via Zomato Platform, the Customer will be entitled to Zomato Pay Benefits (as defined below), subject to successful payment being made by the Customer. For the purposes of Zomato Pay, “Zomato Pay Benefits” shall include but not be limited to either of the following: a. offer(s) including purchasing one item and getting the other of equal or lesser value for free such as 1+1 or 2+2 offers on food and drinks. b. additional banking partner offer(s) applicable on the final payable amount, net of other discounts (excluding tips), provided that the final payable amount meets the minimum and maximum order value criteria for a particular offer, if any, as may be decided by the bank from time to time. scratch card offer(s) with cashback in the form of Zomato Credits which can be used for future dining transactions. Cashback may not always be guaranteed.

# d. The Customers can make a payment for the Bill Amount using Zomato Pay by either scanning the Zomato QR code at the restaurant partnered with Zomato for Zomato Pay or by searching for the restaurant partnered with Zomato for Zomato Pay on the Zomato Platform and selecting Zomato Pay as the payment method. e. The Customer will be solely responsible to pay the restaurant partnered with Zomato for Zomato Pay, the Bill Amount along with all costs and charges payable for all the other items for which you have placed an Order and which are not covered under the Bill Amount. In the event of a concern raised regarding the payment using Zomato Pay, we shall use our best endeavours to assist you however such payment will be subject to verification and confirmation from the restaurant partnered with Zomato for Zomato Pay.

# e. The Customer acknowledges that in order to receive Zomato Credits, the Customer shall scratch the scratch card offer. Once the scratch card offer is revealed to the Customer, the Customer shall avail the Zomato Credits, on future dining transactions only, before the date of expiry of such credits. For the purpose of clarity, the date of expiry of Zomato Credits will be available on each scratch card offer.

# f. Your access to Zomato Pay Benefits shall be subject to receipt of successful payments by Zomato.

# g. The Customer is required to be present at the partner restaurant when using Zomato Pay.

# h. The Customer acknowledges that the Zomato Pay Benefits cannot be clubbed with any ongoing offers by the partner restaurant at the restaurant premise.

# i. The Customer cannot use Zomato Pay for dine in if the Customer is employed at the same partner restaurant.

# j. Zomato reserves the right to terminate / suspend Zomato Pay Benefits to the Customer, if Zomato determines in its sole discretion that: (i) the Customer has violated the terms of Zomato Pay set out herein, (ii) have been involved in activities that are in contravention of the Zomato Pay terms and/or any terms for the usage of Zomato Platform; or (iii) have engaged in activities which are fraudulent / unlawful in nature while availing any of the services of Zomato.

# k. Zomato reserves the right to block and/or terminate/suspend Zomato Pay Benefits on account of breach of these terms including any fraudulent and suspicious activity while using Zomato Pay.

# l. Zomato Pay and the associated Zomato Pay Benefits will be applicable to the Customers on all days. Provided however, the benefit of instant discount(s) applicable on the Bill Value shall not be applicable for the Customer on Exclusion Days (as set out below).

# m. The discounts under Zomato Pay Benefits shall be as follows: a. If a Zomato Gold member uses Zomato Pay via Zomato Platform at a restaurant partnered with Zomato Gold, the Zomato Gold member shall be entitled to a cashback in the form of Zomato Credits, provided the final payable amount is minimum AED 20, b. If a Zomato Gold member uses Zomato Pay via Zomato Platform at a restaurant partnered with Zomato Gold, the Zomato Gold member shall be entitles to special offers such as Buy one Get one, Buy two Get two.

# n. The Customer acknowledges that Zomato Pay is being made available purely on a ‘best effort’ basis and availing Zomato Pay is voluntary.

# o. Zomato reserves the right to modify the Zomato Pay Benefits and/or these Zomato Pay Terms from time to time or at any time, modify or discontinue, temporarily or permanently, Zomato Pay Benefits and/or these Zomato Pay Terms, with or without prior notice and the decision of Zomato shall be final and binding in this regard.

# p. These Zomato Pay Terms do not alter in any way the terms or conditions of any other program or arrangement the Customer may have with Zomato. Termination of Zomato Pay and these Zomato Pay Terms shall have no effect on the Terms of Service governing the contractual relationship between the Customers and Zomato.

# q. For any help or queries, you may reach out to us via chat support or write to us at dining@zomato.com.

# 3. BOOK SERVICE/TABLE RESERVATIONS:
# a. The Customer can make a request for booking a table at a restaurant, offering table reservation via the Zomato Platform and related mobile or software application and such booking will be confirmed to a Customer by email, short message service ("SMS") and/or by any other means of communication only after the restaurant accepts and confirms the booking. The availability of a booking is determined at the time a Customer requests for a table reservation. While using the Zomato Book Service, you shall be required to provide certain details, You agree to provide correct details and warrant that these details are accurate and complete. By submitting a booking request, you express your acceptance to Zomato's terms and privacy policies and agree to receive booking confirmations by email, SMS and/or by any other means of communication after booking a table through the Zomato Book Service. Customer further agrees not to make more than one reservation for Customer's personal use for the same mealtime.

# b. Fees: Zomato may charge booking fee ("Booking Fee") from the Customer upon availing the Zomato Book Service. This Booking Fee shall be adjusted by the restaurant against the total bill for the items consumed by the Customer at such restaurant. Any balance amount remaining to be paid after deduction of the Booking Fee from the restaurant bill shall be payable by the Customer. The Customer shall also be liable to pay any additional charges and/or applicable taxes that may be applicable to the transaction. In the event of any change in the amount of the Booking Fee after the payment is made by the Customer, the amount of the Booking Fee already paid by the Customer will be applicable. The Customer may be required to furnish the payment instrument at the restaurant from which payment has been made for identification purposes.

# c. Modifications & Cancellations: Any request for modification of the confirmed booking will be subject to acceptance of the same by the restaurant. Zomato will use its best endeavours to keep the Customer informed of the status of the booking. For bookings where Booking Fee is not applicable, the Customer may cancel such booking thirty (30) minutes in advance from the scheduled booking time. A confirmed booking for which Booking Fee has been charged from a Customer, modification option will not be available, however the Customer is required to cancel the confirmed booking twenty-four (24) hours prior to the scheduled booking time to avail the refund. Unless otherwise provided herein these Terms, Zomato shall refund the Booking Fee to the Customer within seven (7) working days from the date of such cancellation. However, Zomato reserves the right to retain the Booking Fee in the event the Customers fails to cancel the booking within the estimated timeframe mentioned herein above.

# d. Late Arrivals: Zomato advises the Customer to arrive 10 minutes in advance of the scheduled booking time. The restaurant reserves the right to cancel your booking and allocate the table to other guests in case of late arrivals and Zomato shall in no manner be liable for such cancellation initiated by the Restaurant. Zomato hereby reserves its right to retain the Booking Fee paid by the Customers, in the event the Customer is late by more than 10 minutes from the scheduled booking time and/or fails to show up at the restaurant.

# e. Dispute: In the event the restaurant fails to honour the confirmed booking or in case of any other complaint or dispute raised by the Customer in relation to the booking, the Customer shall raise such disputes with Zomato within 30 minutes from the scheduled booking time at the helpline numbers as provided herein below. Upon receipt of such complaint or dispute, Zomato will make reasonable efforts to verify the facts of such complaint/ dispute with the restaurant and may at its sole discretion initiate the refund of the Booking Fee to such Customer.

# f. Personal Information: Customers will be required to share certain personal information with Zomato and/or the restaurant including but not limited to their name, phone number, email address in order to avail the Zomato Book Service and the Customer hereby permits Zomato to share such personal information with the restaurant for confirming such Customer's booking and/or such other communication relating to but not limited to the Zomato Book Service or any promotions by the restaurant. Zomato will use these details in accordance with the Privacy Policy published here. Zomato will share your personal information with the restaurant for the purpose of your reservation. However, notwithstanding anything otherwise set out herein, Zomato shall in no manner be liable for any use of your personal information by such restaurant for any purpose whatsoever.

# g. Additional Request: In the case of any additional request communicated by the Customer at the time of the booking, the same will be conveyed to the restaurant by Zomato and confirmed to the Customer basis restaurant's response. While Zomato will take all the care to ensure timely communication of these requests to both the Customer and the restaurant, the liability to fulfill the request lies solely with the restaurant and Zomato shall in no manner be liable if the restaurant does not honor any of the confirmed additional requests of the Customers.

# h. Call Recording: Zomato may contact via telephone, SMS or other electronic messaging or by email with information about your Zomato Book Service or any feedback thereon. Any calls that may be made by Zomato, by itself or through a third party, to the Customers or the restaurant pertaining to any booking requests of a Customer may be recorded for internal training and quality purposes by Zomato or any third party appointed by Zomato.

# i. Liability Limitation: Notwithstanding anything otherwise set out herein, Zomato shall in no manner be liable in any way for any in-person interactions with the restaurant as a result of the booking or for the Customer's experience at the restaurant or in the event a restaurant does not honor a confirmed booking. Zomato is only a platform connecting Customers to the restaurant and shall not be liable for any acts or omissions on part of the restaurant including deficiency in service, quality of food, time taken to serve or any other experience of the Customer.

# j. Contact Us: You may write to us at bookings@zomato.com for any further queries with regard to the Zomato Book Service and may also contact us on the following numbers for more information: For India: 011-33107581; For UAE: 04-4376083;

# 4. ZOMATO GOLD:
# For Zomato Gold members in India:

# Zomato Gold is an invite-only membership based program available in select cities which allows its members to avail Offer(s) (as defined below), on dine in and food ordering and delivery offered by a host of restaurants partnered with Zomato ("Zomato Gold Partner Restaurants").

# 1. Zomato Gold Membership:

# As a member of Zomato Gold, You will be entitled to avail Offer(s) on the Bill Value (as defined below), provided that the Bill Value for such order is above the minimum order value (if applicable) as determined by the Zomato Gold Partner Restaurant when You pay the Bill Value via the Zomato Platform based on and subject to the membership plan purchased by You via the Zomato Platform.

# For the purpose of Zomato Gold, “Bill Value” shall mean the total amount set out in the bill for food and beverages availed by the member at the Zomato Gold Partner Restaurants, and shall not include applicable taxes, Delivery Charges, service charge and other charges as may be applicable and “Offer(s)” shall include, but not be restricted to, either (i) percentage of discount or flat discount that the Zomato Gold Partner Restaurant agrees to extend to the Customer on the Bill Value for each transaction; (ii) meal packages that are fixed price deals offered by the Zomato Gold Partner Restaurant; and/or (iii) other benefits and offers as may be communicated on the Zomato Platform.

# 2. Benefits and features under Zomato Gold membership:

# i. Offer(s) can be redeemed at Zomato Gold Partner Restaurants only and the list of such Zomato Gold Partner Restaurants may be updated periodically;
# ii. Offer(s) may be changed or added from time to time. You are advised to check the Offer(s) being offered by the Zomato Gold Partner Restaurant at the time of placing your order;
# iii. Offer(s) cannot be exchanged for cash.
# iv. Offer(s) can only be availed in the cities in which Zomato Gold is available.
# v. Offer(s) shall be extended only if the Zomato Gold member makes payment towards the Bill Value via the Zomato Platform.
# vi. You will be responsible to pay the Zomato Gold Partner Restaurants all costs and charges payable for all the other items for which you have placed an order and which are not covered under the Offer(s).
# vii. There is no limit on the number of times the Offer(s) can be availed in a day.
# viii. You are not permitted to avail the Offer(s) on more than two (2) devices at a time.
# ix. Zomato Gold Partner Restaurants offering Zomato Gold for delivery may differ from Zomato Gold Partner Restaurants offering Zomato Gold for dine out.
# x. The term of Your Zomato Gold membership shall be subject to the membership plan opted by You.

# 3. Zomato Gold for Dine in:

# While availing Zomato Gold benefits on dine-in, following terms and conditions shall be specifically applicable to Zomato Gold members:

# a. Upon fulfilment of payment of the Bill Value via the Zomato Platform, You will be required to show the payment confirmation to the Zomato Gold Partner Restaurant.
# b. Offer(s) cannot be clubbed during the same visit.
# c. Zomato Gold members are required to be present at the Zomato Gold Partner Restaurant when availing the Offer(s).
# d. Offer(s) cannot be clubbed with any ongoing Zomato Gold Partner Restaurant Offer(s) or on menu items which are being sold on discount except items sold on maximum retail price (MRP).
# e. Offer(s) extended by the Zomato Gold Partner Restaurant shall be valid irrespective of the number of people seated on the table.
# f. Offer(s) shall not be applicable on tobacco and related products.
# g. Customer cannot use Zomato Gold for dine-in if the Customer is employed at the same Zomato Gold Partner Restaurant.

# 4. Zomato Gold for Food Ordering and Delivery:

# While availing Zomato Gold benefits on food ordering and delivery, following terms and conditions shall be specifically applicable to Zomato Gold members:

# a. As a member of Zomato Gold, you can avail Offer(s) on order placed by You from Zomato Gold Partner Restaurants.
# b. Full discount on Delivery Charges (i.e. Delivery Charges including any base fee, surge fee, and distance fee) under Zomato Gold membership is (a) only applicable on restaurants which are located under a certain distance (“Distance”) from you, and this Distance is communicated while You place the order. The Distance is calculated as per the estimated Distance that will be travelled by the Delivery Partner to deliver your order from the restaurant to your delivery location. Since this Distance is calculated using third-party sources, Zomato does not hold any liability on the accuracy of this data and (b) may not be applicable in cases where the orders from the restaurants are cash on delivery orders. (c) only applicable above a certain order value, which will be communicated to you while you place the order. (d) not applicable on select restaurants. c. The Offer(s) can be clubbed with any other offers or discounts or deals extended by the Zomato Gold Partner Restaurant or Zomato or any other third party.
# d. The Offer(s) is not valid on menu items sold by the Zomato Gold Partner Restaurant at maximum retail price (MRP), combos and any other items being sold at discount by the Zomato Gold Partner Restaurant.
# e. The Offer(s) can be availed only for orders placed for home delivery.
# f. Selective orders at certain Restaurant Partners may be eligible for On Time Guarantee benefit. The terms and conditions mentioned below for On Time Guarantee will be applicable to such selective orders.

# 5. The Zomato Gold Partner Restaurants may change from time to time and the members are advised to keep a check on the updated list of Zomato Gold Partner Restaurants from time to time at Zomato’s Platform.

# 6. Fees: In order to avail Zomato Gold membership, members are required to pay a membership fee which shall be based on the membership plan opted by such member.

# 7. Payments: To purchase and/or renew your membership plan, you can choose a payment method, as available on the 'Payment' section of the Zomato Platform. Your access to Zomato Gold shall be subject to receipt of successful payments by Zomato. The membership fee shall be inclusive of all applicable taxes. For some payment methods, the issuer may charge you certain fees, such as foreign transaction fees or other fees, relating to the processing of your payment method. Zomato shall require additional information and/or documents from you in accordance to the applicable local laws in your or as per the internal requirements of Zomato.

# 8. Terms: These Terms will begin on the date of purchase of the membership plan and will be valid till such time your membership plan expires.

# 9. Subscription and Renewal: The membership, once purchased, is non-transferable and non-refundable.

# 10. Modification to Zomato Gold: Zomato reserves the right to offer, alter, extend or withdraw, as the case may be, any offers or discounts or promotions extended by Zomato at any time with or without giving any prior notice to the Customer. In such cases, such revision will be updated on the Zomato Platform accordingly.

# 11. Zomato reserves the right to terminate / suspend Your membership to the Zomato Gold , if Zomato determines in its sole discretion that (i) You have violated the terms and conditions of Zomato Gold set out herein, (ii) have been involved in activities that are in contravention of the Zomato Gold terms and/or any terms for the usage of Zomato Platform; or (iii) have engaged in activities which are fraudulent/unlawful in nature while availing any of Services of Zomato. You will not be eligible for any refund if the Zomato Gold membership has been terminated/suspended by Zomato for such cases.

# 12. Exclusion Days: All benefits of Zomato Gold can be used on any day of the week except on the days listed herein below ("Exclusion Days"). Exclusion period applies from the start of operational hours for the day up till 6 a.m. of the following day. In addition to the Exclusion Days mentioned herein, Zomato Gold shall not be extended by Zomato Gold Partner Restaurant on the days prohibited by law. Zomato Gold Partner Restaurant, at its discretion, may or may not extend Zomato Gold benefits to the Customer on an Exclusion Day. Zomato Gold members are advised to make prior enquiry with the Zomato Gold Partner Restaurants and/or check the Zomato Platform before ordering and/or visiting the Zomato Gold Partner Restaurant, to confirm whether the Zomato Gold can be redeemed or not. Zomato reserves the right to add Exclusion Days to Zomato Gold at its discretion which will be updated on Zomato Platform from time to time.

# No.	DATE	EVENT
# 1	December 31 and January 1	New Year's Eve/Day
# 2	As per lunar calendar	Pongal (applicable in Tamil Nadu only)
# 3	February 14	Valentine's Day
# 4	As per lunar calendar	Durga Puja (applicable in West Bengal only)
# 5	As per lunar calendar	Diwali Eve/Day
# 6	December 24 and December 25	Christmas Eve/Day
# 13. Personal Information: Zomato Gold members will be required to share certain personal information with Zomato including their name, phone number, email address, payment details, in order to purchase Zomato Gold. Zomato will use these details in accordance with the Privacy Policy published on www.zomato.com/privacy.

# 14. Disclaimer: The liability to extend the benefits under Zomato Gold rests solely with the Zomato Gold Partner Restaurants and Zomato shall in no manner be liable if the Zomato Gold Partner Restaurants does not honour the benefits under Zomato Gold. Zomato Gold Partner Restaurants reserve the right to refuse service to anyone in accordance with its policies. However, in the event a Zomato Gold Partner Restaurant refuses to honour Zomato Gold in accordance with these Terms, please reach us via Gold chat support on the Zomato app or write to us at gold@zomato.com and we shall use our best endeavour to assist you.

# 15. The Customer acknowledges that Zomato bears no responsibility for the compliance with statutory rules, regulations and licences by Zomato Gold Partner Restaurant. The Customer agrees that Zomato shall not be liable in any manner if the Customer is unable to avail the benefits under Zomato Gold with a Zomato Gold Partner Restaurant due to the Zomato Gold Partner Restaurant’s violation of any statutory rule, regulation and licence.

# 16. Liability Limitation: Notwithstanding anything otherwise set out herein, Zomato shall in no manner be liable in any way for any in-person interactions with representatives or staff of the Zomato Gold Partner Restaurant or for the member’s experience at the Zomato Gold Partner Restaurant. Zomato shall in no manner be liable to the member if any outlet of Zomato Gold Partner Restaurant temporarily or permanently shuts down its operations. Notwithstanding anything set out herein, Zomato’s aggregate liability for any or all claims arising from or in connection with your use of Zomato Gold shall be limited to the membership fee paid by you at the time of purchasing the membership.

# 17. Call Recording: Zomato may contact Zomato Gold members via telephone, SMS or other electronic messaging or by email with information about your Zomato Gold experience or any feedback thereon. Any calls that may be made by Zomato, by itself or through a third party, to the members or the restaurant pertaining to the experience of a Customer may be recorded for internal training and quality purposes by Zomato or any third party appointed by Zomato.

# 18. Assignment: Zomato may assign or transfer any of its rights or obligations under these Terms and conditions to any of its affiliates or any third party at any time.

# 19. Contact Us: You may contact us at gold@zomato.com for any further queries with regard to Zomato Gold.

# For Zomato Gold members in UAE:

# Zomato Gold is a membership based program available in all cities which allows its members to avail Offer(s) (as defined below), on dine in offered by a host of restaurants partnered with Zomato ("Zomato Gold Partner Restaurants").

# 1. Zomato Gold Membership:

# As a member of Zomato Gold, You will be entitled to avail Offer(s) on the Bill Value (as defined below), provided that the Bill Value for such order is above the minimum order value (if applicable) as determined by the Zomato Gold Partner Restaurant when You pay the Bill Value via the Zomato Platform based on and subject to the membership plan purchased by You via the Zomato Platform.

# For the purpose of Zomato Gold, “Bill Value” shall mean the total amount set out in the bill for food and beverages availed by the member at the Zomato Gold Partner Restaurants, and shall not include applicable taxes, Delivery Charges, service charge and other charges as may be applicable and “Offer(s)” shall include, but not be restricted to, either (i) customer credits, promo codes, vouchers or any other form of cashback that Zomato may decide as it’s discretion; (ii) meal packages that are fixed price deals offered by the Zomato Gold Partner Restaurant; and/or (iii) other benefits and offers as may be communicated on the Zomato Platform (iv) flat % off or as may be communicated while paying the dining bill on the Zomato Platform (v) Unlocks Offers which include purchasing one time and getting another item of the equal or lesser value free such as 1+1 or 2+2 offers on food and drinks (vi) or any such Offer(s) as maybe communicated to the Customers by Zomato.

# 2. Benefits and features under Zomato Gold membership:

# i. Offer(s) can be redeemed at Zomato Gold Partner Restaurants only and the list of such Zomato Gold Partner Restaurants may be updated periodically;
# ii. Offer(s) may be changed or added from time to time. You are advised to check the Offer(s) being offered by the Zomato Gold Partner Restaurant at the time of placing your order;
# iii. Offer(s) cannot be exchanged for cash.
# iv. Offer(s) shall be extended only if the Zomato Gold member makes payment towards the Bill Value via the Zomato Platform.
# v. You will be responsible to pay the Zomato Gold Partner Restaurants all costs and charges payable for all the other items for which you have placed an order and which are not covered under the Offer(s).
# vi. There is no limit on the number of times the Offer(s) can be availed in a day.
# viii. You are not permitted to avail the Offer(s) on more than two (2) devices at a time.
# ix. The term of Your Zomato Gold membership shall be subject to the membership plan opted by You.

# 3. Zomato Gold for Dine in:

# While availing Zomato Gold benefits on dine-in, following terms and conditions shall be specifically applicable to Zomato Gold members:

# a. Upon fulfilment of payment of the Bill Value via the Zomato Platform, You will be required to show the payment confirmation to the Zomato Gold Partner Restaurant.
# b. Offer(s) cannot be clubbed during the same visit.
# c. Zomato Gold members are required to be present at the Zomato Gold Partner Restaurant when availing the Offer(s).
# d. Offer(s) cannot be clubbed with any ongoing Zomato Gold Partner Restaurant Offer(s) or on menu items which are being sold on discount.
# e. Offer(s) extended by the Zomato Gold Partner Restaurant shall be valid irrespective of the number of people seated on the table except for unlock offer(s) where a minimum of two Customers are required to avail the Offer(s).

# f. Offer(s) shall not be applicable on tobacco and related products.
# g. Customer cannot use Zomato Gold for dine-in if the Customer is employed at the same Zomato Gold Partner Restaurant.

# 4. The Zomato Gold Partner Restaurants may change from time to time and the members are advised to keep a check on the updated list of Zomato Gold Partner Restaurants from time to time at Zomato’s Platform.

# 5. Fees: In order to avail Zomato Gold membership, members are required to pay a membership fee which shall be based on the membership plan opted by such member.

# 6. Payments: To purchase and/or renew your membership plan, you can choose a payment method, as available on the 'Payment' section of the Zomato Platform. Your access to Zomato Gold shall be subject to receipt of successful payments by Zomato. The membership fee shall be inclusive of all applicable taxes. For some payment methods, the issuer may charge you certain fees, such as foreign transaction fees or other fees, relating to the processing of your payment method. Zomato shall require additional information and/or documents from you in accordance to the applicable local laws in your or as per the internal requirements of Zomato.

# 7. Terms: These Terms will begin on the date of purchase of the membership plan and will be valid till such time your membership plan expires.

# 8. Subscription and Renewal: The membership, once purchased, is non-transferable and non-refundable.

# 9. Modification to Zomato Gold: Zomato reserves the right to offer, alter, extend or withdraw, as the case may be, any offers or discounts or promotions extended by Zomato at any time with or without giving any prior notice to the Customer. In such cases, such revision will be updated on the Zomato Platform accordingly.

# 10. Zomato reserves the right to terminate / suspend Your membership to the Zomato Gold, if Zomato determines in its sole discretion that (i) You have violated the terms and conditions of Zomato Gold set out herein, (ii) have been involved in activities that are in contravention of the Zomato Gold terms and/or any terms for the usage of Zomato Platform; or (iii) have engaged in activities which are fraudulent/unlawful in nature while availing any of Services of Zomato. You will not be eligible for any refund if the Zomato Gold membership has been terminated/suspended by Zomato for such cases.

# 11. Personal Information: Zomato Gold members will be required to share certain personal information with Zomato including their name, phone number, email address, payment details, in order to purchase Zomato Gold. Zomato will use these details in accordance with the Privacy Policy published on www.zomato.com/privacy.

# 12. Disclaimer: The liability to extend the benefits under Zomato Gold rests solely with the Zomato Gold Partner Restaurants and Zomato shall in no manner be liable if the Zomato Gold Partner Restaurants does not honour the benefits under Zomato Gold. Zomato Gold Partner Restaurants reserve the right to refuse service to anyone in accordance with its policies. However, in the event a Zomato Gold Partner Restaurant refuses to honour Zomato Gold in accordance with these Terms, please reach us via Gold chat support on the Zomato app or write to us at gold@zomato.com and we shall use our best endeavour to assist you.

# 13. The Customer acknowledges that Zomato bears no responsibility for the compliance with statutory rules, regulations and licences by Zomato Gold Partner Restaurant. The Customer agrees that Zomato shall not be liable in any manner if the Customer is unable to avail the benefits under Zomato Gold with a Zomato Gold Partner Restaurant due to the Zomato Gold Partner Restaurant’s violation of any statutory rule, regulation and licence.

# 14. Liability Limitation: Notwithstanding anything otherwise set out herein, Zomato shall in no manner be liable in any way for any in-person interactions with representatives or staff of the Zomato Gold Partner Restaurant or for the member’s experience at the Zomato Gold Partner Restaurant. Zomato shall in no manner be liable to the member if any outlet of Zomato Gold Partner Restaurant temporarily or permanently shuts down its operations. Notwithstanding anything set out herein, Zomato’s aggregate liability for any or all claims arising from or in connection with your use of Zomato Gold shall be limited to the membership fee paid by you at the time of purchasing the membership.

# 15. Call Recording: Zomato may contact Zomato Gold members via telephone, SMS or other electronic messaging or by email with information about your Zomato Gold experience or any feedback thereon. Any calls that may be made by Zomato, by itself or through a third party, to the members or the restaurant pertaining to the experience of a Customer may be recorded for internal training and quality purposes by Zomato or any third party appointed by Zomato.

# 16. Assignment: Zomato may assign or transfer any of its rights or obligations under these Terms and conditions to any of its affiliates or any third party at any time.

# 17. Contact Us: You may contact us at gold@zomato.com for any further queries with regard to Zomato Gold.

# 5. FOOD HYGIENE RATINGS:
# a. The Food Hygiene Ratings ("Hygiene Rating(s)") is an initiative of Zomato in partnership with certified auditors ("Hygiene Auditor(s)") to audit restaurants. The Customer acknowledges that Zomato is merely acting as a facilitator in the hygiene audit process and does not conduct any hygiene audit by itself.

# b. The Customer understands and agrees that the Hygiene Rating(s) displayed on the Zomato Platform are for informational purposes only and merely indicate the hygiene standards of a restaurant at the time such audit is conducted by the Hygiene Auditor(s). The Hygiene Rating(s) shall not be deemed to be an indicator to the food quality standards maintained by a restaurant.

# c. Validity:
# i. The validity of the Hygiene Rating(s) displayed on the Zomato Platform shall be for a period of six (6) or twelve (12) months, as the case may be, from the date of last audit as displayed on the Zomato Platform.
# ii. Zomato reserves the right to remove the Hygiene Rating(s) for a restaurant upon expiry of the validity of the Hygiene Rating(s), without any prior intimation to the Customer.

# d. Disclaimer and Liability:
# i. The Hygiene Rating(s) that are displayed on the Zomato Platform are on an 'as available' basis, based on the data provided to Zomato by the Hygiene Auditor(s) for a restaurant and Zomato disclaims all warranties with respect to the Hygiene Rating(s) or any information displayed in this regard on the Zomato Platform.
# ii. Any actions taken by a Customer relying upon the Hygiene Rating(s) or any information displayed in this regard on the Zomato Platform shall be strictly at such Customer's own risk and Zomato shall in no manner be held liable for any losses or damages that may arise in connection with the use of this information or any inaccuracy, invalidity or discrepancy in the Hygiene Rating(s). Zomato expressly disclaims all liabilities that may arise in connection to the reliance by a Customer on such Hygiene Rating(s) including without limitation, any consumption of food or any other items served at a restaurant, or any other services that may be provided by a restaurant.
# iii. Zomato shall under no circumstances be held liable if a restaurant does not display the correct and accurate Hygiene Rating(s) on its restaurant premises, website or any other platform.

# e. The Customer acknowledges that the Hygiene Rating(s) as displayed on the Zomato Platform shall under no circumstances be construed to be a proof of the hygiene standards or practices that are being adopted by the restaurant and such Hygiene Rating(s) shall not be used as evidence in a court of law or governmental authority or disputed in any manner whatsoever. The Customer further understands that the restaurant is solely responsible to maintain the hygiene and food safety standards in compliance with the applicable laws.

# f. Contact Us: You may contact us at hygiene@zomato.com for any further queries with regard to Hygiene Ratings.

# 6. EDITION WALLET:
# a. Edition Wallet (as defined below) is an initiative of Zomato in partnership with RBL Bank ("Banking Partner") to facilitate digital payments through the Zomato Platform.

# For the purposes of these terms and conditions, "Edition Wallet" shall mean the account opened by the Customer on the Zomato Platform, maintained with the Banking Partner in accordance with the Banking Partner Terms and Conditions and Conditions (as defined below).

# b. Edition Wallet is presently only available for Customers in India.

# c. The Customer acknowledges that the Customer shall comply with the RBL Reloadable Prepaid Wallet Terms and Conditions, available at https://www.zomato.com/policies/wallet-terms-and-conditions/, issued by the Banking Partner governing the issue and use of the Edition Wallet ("Banking Partner Terms and Conditions").

# d. In order to register, create, use, and close the Edition Wallet, Zomato may require the Customer to submit certain personal information and documents including but not limited to Customer’s name, address, mobile phone number, e-mail address, address proof, PAN Card, driving license etc. to the Banking Partner through the Zomato Platform, as may be required in accordance with the Banking Partner Terms and Conditions. Zomato will use these details in accordance with the Privacy Policy published at www.zomato.com/privacy.

# e. The Customer acknowledges that the KYC details will be verified by the Banking Partner prior to issuance of Edition Wallet and issuance of Edition Wallet shall be at the sole discretion of the Banking Partner.

# f. The Edition Wallet can be used by the Customer only on the Zomato mobile application, to make payments for purchases including orders, tipping Delivery Partners, Pro/Pro Plus membership purchase/renewals, dining out payments at Partner Restaurants, subject to availability of funds in the Customer’s Edition Wallet.

# g. The Customer shall be entitled to offers and/or benefits in relation to the use of Edition Wallet which may be communicated by Zomato from time to time. Such offers and/or benefits shall be subject to specific terms and conditions and may vary from offer to offer.

# h. The Customer’s Edition Wallet shall have a limit on addition and a balance limit as communicated to the Customer from time to time.

# i. Zomato reserves the right, to offer, alter, extend or withdraw, as the case may be, the Edition Wallet feature or any offer associated with the Edition Wallet extended by Zomato at any time with or without giving any prior notice to the Customers. In such cases, such revision will be updated on the Zomato Platform accordingly.

# j. A Customer can cancel the Edition Wallet by raising a request for closure via chat support. Upon receipt of such request for closure, the Edition Wallet will be permanently closed and accordingly the associated offers shall be revoked. For clarity, once the Edition Wallet is permanently closed, the same cannot be reactivated.

# k. In the event the Customer cancels the Edition Wallet at any time during the validity of the Edition Wallet, the Customer will be eligible to receive a refund, as applicable at the time of such request, in the source account (based on the Wallet balance on the day of refund initiation) in accordance with the Banking Partner Terms and Conditions and the applicable RBI guidelines. Upon cancellation of the Wallet, the Customer will lose the Zomato Pro membership and the associated offers.

# l. Contact Us: For any further query with regard to Edition Wallet, you may contact us via in app chat support or write to us at wallet@zomato.com.

# 7. DELIVERY OFFER / PLAN:
# a. Delivery Offer / Plan is a limited-time, invite-only offer whereby You can avail discounted delivery for the next few orders ("Offer"). You can claim the Offer by opting for it while placing an order on the Zomato platform, where You will be required to pay an additional amount to avail the Offer ("Claim Order"). Once the Claim Order is successfully placed, the benefits under the Offer will be made available to You, to be used on your subsequent eligible orders, in accordance with the terms and conditions set out below.

# b. In the event You avail the Offer, the following Terms and Conditions shall be applicable to You:

# i. Under the Offer, You will be entitled to discounted delivery, including any distance fee or delivery surge, for the subsequent eligible orders, as may be communicated to You on the Zomato platform.

# ii. This Offer may provide a limited number of discounted deliveries ("Count"). Such Count shall be communicated to You on the Zomato platform, while claiming the Offer.

# iii. Once the Count(s) are exhausted under the Offer, the Offer shall lapse.

# iv. The Offer is available for a limited duration, as may be communicated to You on the Zomato platform, while claiming the Offer ("Offer Period"). After the expiry of Offer Period, the Offer shall lapse.

# v. The Offer shall only be applicable in India and can be availed in all cities in India where the customer has the option to place an order on the Zomato platform, for delivery.

# vi. The Offer is applicable only for all online paid orders for delivery and not shall not be applicable on other orders, such as take-away, dine-in order, grocery orders etc.

# vii. The Offer is applicable on all restaurants.

# viii. The Offer may be applicable on orders placed with the restaurants having a minimum order value, as may be communicated to You while availing the Offer.

# ix. The Offer shall not be applicable on inter-city orders and such other services as may be communicated to You on the Zomato platform.

# x. The Offer cannot be exchanged for cash or any other receivables.

# xi. The Offer once claimed, is auto-applied on the subsequent eligible orders within the Offer Period. For clarity, You cannot remove the Offer from certain orders.

# xii. You will not be eligible for any refund or compensation after opting for the Offer.

# xiii. The Offer is non-transferable, non refundable once claimed. If the order in which the Offer was claimed, is cancelled, the Offer will stand cancelled along with it. Any refund applicable to the order will be derived as per Zomato's policies.

# xiv. Zomato reserves the right to block a device and/or terminate/suspend the Offer on account of breach of these terms and conditions including any fraudulent and suspicious activity while availing the Offer or availing the Offer.

# xv. Zomato reserves the right to offer, alter, extend or withdraw, as the case may be, any Offer(s) or offers or discounts or promotions extended by Zomato at any time by providing prior notice to You. In such cases, such revision will be updated on the Zomato Platform accordingly.

# xvi. Contact Us: You may contact us via chat support on the Zomato app or email us at order@zomato.com for any further queries with regard to the Offer.

# XIV. Disclaimer of warranties, limitation of liability, and Indemnification
# 1. Disclaimer of Warranties
# YOU ACKNOWLEDGE AND AGREE THAT THE SERVICES ARE PROVIDED "AS IS" AND "AS AVAILABLE" AND THAT YOUR USE OF THE SERVICES SHALL BE AT YOUR SOLE RISK. TO THE FULLEST EXTENT PERMITTED BY APPLICABLE LAW, ZOMATO, ITS AFFILIATES AND THEIR RESPECTIVE OFFICERS, DIRECTORS, EMPLOYEES, AGENTS, AFFILIATES, BRANCHES, SUBSIDIARIES, AND LICENSORS ("ZOMATO PARTIES") DISCLAIM ALL WARRANTIES, EXPRESS OR IMPLIED, IN CONNECTION WITH THE SERVICES INCLUDING MOBILE APPS AND YOUR USE OF THEM. TO THE FULLEST EXTENT PERMITTED BY APPLICABLE LAW, THE ZOMATO PARTIES MAKE NO WARRANTIES OR REPRESENTATIONS THAT THE SERVICES HAVE BEEN AND WILL BE PROVIDED WITH DUE SKILL, CARE AND DILIGENCE OR ABOUT THE ACCURACY OR COMPLETENESS OF THE SERVICES' CONTENT AND ASSUME NO RESPONSIBILITY FOR ANY (I) ERRORS, MISTAKES, OR INACCURACIES OF CONTENT, (II) PERSONAL INJURY OR PROPERTY DAMAGE, OF ANY NATURE WHATSOEVER, RESULTING FROM YOUR ACCESS TO AND USE OF THE SERVICES, (III) ANY UNAUTHORIZED ACCESS TO OR USE OF OUR SERVERS AND/OR ANY AND ALL PERSONAL INFORMATION STORED THEREIN, (IV) ANY INTERRUPTION OR CESSATION OF TRANSMISSION TO OR FROM THE SERVICES, (V) ANY BUGS, VIRUSES, TROJAN HORSES, OR THE LIKE WHICH MAY BE TRANSMITTED TO OR THROUGH THE SERVICES THROUGH THE ACTIONS OF ANY THIRD PARTY, (VI) ANY LOSS OF YOUR DATA OR CONTENT FROM THE SERVICES AND/OR (VII) ANY ERRORS OR OMISSIONS IN ANY CONTENT OR FOR ANY LOSS OR DAMAGE OF ANY KIND INCURRED AS A RESULT OF THE USE OF ANY CONTENT POSTED, EMAILED, TRANSMITTED, OR OTHERWISE MADE AVAILABLE VIA THE SERVICES. ANY MATERIAL DOWNLOADED OR OTHERWISE OBTAINED THROUGH THE USE OF THE SERVICES IS DONE AT YOUR OWN DISCRETION AND RISK AND YOU WILL BE SOLELY RESPONSIBLE FOR ANY DAMAGE TO YOUR COMPUTER SYSTEM OR OTHER DEVICE OR LOSS OF DATA THAT RESULTS FROM THE DOWNLOAD OF ANY SUCH MATERIAL. THE ZOMATO PARTIES WILL NOT BE A PARTY TO OR IN ANY WAY BE RESPONSIBLE FOR MONITORING ANY TRANSACTION BETWEEN YOU AND THIRD-PARTY PROVIDERS OF PRODUCTS OR SERVICES. YOU ARE SOLELY RESPONSIBLE FOR ALL OF YOUR COMMUNICATIONS AND INTERACTIONS WITH OTHER CUSTOMERS OF THE SERVICES AND WITH OTHER PERSONS WITH WHOM YOU COMMUNICATE OR INTERACT AS A RESULT OF YOUR USE OF THE SERVICES. NO ADVICE OR INFORMATION, WHETHER ORAL OR WRITTEN, OBTAINED BY YOU FROM ZOMATO OR THROUGH OR FROM THE SERVICES SHALL CREATE ANY WARRANTY NOT EXPRESSLY STATED IN THE TERMS. UNLESS YOU HAVE BEEN EXPRESSLY AUTHORIZED TO DO SO IN WRITING BY ZOMATO, YOU AGREE THAT IN USING THE SERVICES, YOU WILL NOT USE ANY TRADE MARK, SERVICE MARK, TRADE NAME, LOGO OF ANY COMPANY OR ORGANIZATION IN A WAY THAT IS LIKELY OR INTENDED TO CAUSE CONFUSION ABOUT THE OWNER OR AUTHORIZED USER OF SUCH MARKS, NAMES OR LOGOS.

# 2. Limitation of Liability
# TO THE FULLEST EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL THE ZOMATO PARTIES BE LIABLE TO YOU FOR ANY DAMAGES RESULTING FROM ANY (I) ERRORS, MISTAKES, OR INACCURACIES OF CONTENT, AND/OR (II) PERSONAL INJURY OR PROPERTY DAMAGE, OF ANY NATURE WHATSOEVER, RESULTING FROM YOUR ACCESS TO AND USE OF THE SERVICES INCLUDING MOBILE APP, AND/OR (III) ANY UNAUTHORIZED ACCESS TO OR USE OF OUR SERVERS AND/OR ANY AND ALL PERSONAL INFORMATION STORED THEREIN, AND/OR (IV) ANY INTERRUPTION OR CESSATION OF TRANSMISSION TO OR FROM OUR SERVERS, AND/OR (V) ANY BUGS, VIRUSES, TROJAN HORSES, OR THE LIKE, WHICH MAY BE TRANSMITTED TO OR THROUGH THE SERVICES BY ANY THIRD PARTY, AND/OR (VI) ANY LOSS OF YOUR DATA OR CONTENT FROM THE SERVICES, AND/OR (VII) ANY ERRORS OR OMISSIONS IN ANY CONTENT OR FOR ANY LOSS OR DAMAGE OF ANY KIND INCURRED AS A RESULT OF YOUR USE OF ANY CONTENT POSTED, TRANSMITTED, OR OTHERWISE MADE AVAILABLE VIA THE SERVICES, WHETHER BASED ON WARRANTY, CONTRACT, TORT, OR ANY OTHER LEGAL THEORY, AND WHETHER OR NOT THE ZOMATO PARTIES ARE ADVISED OF THE POSSIBILITY OF SUCH DAMAGES, AND/OR (VIII) THE DISCLOSURE OF INFORMATION PURSUANT TO THESE TERMS OR OUR PRIVACY POLICY, AND/OR (IX) YOUR FAILURE TO KEEP YOUR PASSWORD OR ACCOUNT DETAILS SECURE AND CONFIDENTIAL, AND/OR (X) LOSS OR DAMAGE WHICH MAY BE INCURRED BY YOU, INCLUDING BUT NOT LIMITED TO LOSS OR DAMAGE AS A RESULT OF RELIANCE PLACED BY YOU ON THE COMPLETENESS, ACCURACY OR EXISTENCE OF ANY ADVERTISING, OR AS A RESULT OF ANY RELATIONSHIP OR TRANSACTION BETWEEN YOU AND ANY ADVERTISER OR SPONSOR WHOSE ADVERTISING APPEARS ON THE SERVICES, AND/OR DELAY OR FAILURE IN PERFORMANCE RESULTING FROM CAUSES BEYOND ZOMATO'S REASONABLE CONTROL. IN NO EVENT SHALL THE ZOMATO PARTIES BE LIABLE TO YOU FOR ANY INDIRECT, INCIDENTAL, SPECIAL, PUNITIVE, EXEMPLARY OR CONSEQUENTIAL DAMAGES WHATSOEVER, HOWEVER CAUSED AND UNDER ANY THEORY OF LIABILITY, INCLUDING BUT NOT LIMITED TO, ANY LOSS OF PROFIT (WHETHER INCURRED DIRECTLY OR INDIRECTLY), ANY LOSS OF GOODWILL OR BUSINESS REPUTATION, ANY LOSS OF DATA SUFFERED, COST OF PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES, OR OTHER INTANGIBLE LOSS.

# Indemnification
# You agree to indemnify, defend, and hold harmless the Zomato Parties from and against any third party claims, damages (actual and/or consequential), actions, proceedings, demands, losses, liabilities, costs and expenses (including reasonable legal fees) suffered or reasonably incurred by us arising as a result of, or in connection with: (i) Your Content, (ii) your unauthorized use of the Services, or products or services included or advertised in the Services; (iii) your access to and use of the Services; (iv) your violation of any rights of another party; or (v) your breach of these Terms, including, but not limited to, any infringement by you of the copyright or intellectual property rights of any third party. We retain the exclusive right to settle, compromise and pay, without your prior consent, any and all claims or causes of action which are brought against us. We reserve the right, at your expense, to assume the exclusive defense and control of any matter for which you are required to indemnify us and you agree to cooperate with our defense of these claims. You agree not to settle any matter in which we are named as a defendant and/or for which you have indemnity obligations without our prior written consent. We will use reasonable efforts to notify you of any such claim, action or proceeding upon becoming aware of it.

# XV. Termination of your access to the services
# You can delete your account at any time by contacting us via the "Contact Us" link at the bottom of every page or by following this process: Go to Profile > Setting > Security > click on the 'Delete Account' button and ceasing further use of the Services.

# We may terminate your use of the Services and deny you access to the Services in our sole discretion for any reason or no reason, including your: (i) violation of these Terms; or (ii) lack of use of the Services. You agree that any termination of your access to the Services may be affected without prior notice, and acknowledge and agree that we may immediately deactivate or delete your account and all related information and/or bar any further access to your account or the Services. If you use the Services in violation of these Terms, we may, in our sole discretion, retain all data collected from your use of the Services. Further, you agree that we shall not be liable to you or any third party for the discontinuation or termination of your access to the Services

# XVI. General terms
# Interpretation:

# The section and subject headings in these Terms are included for reference only and shall not be used to interpret any provisions of these Terms.

# Entire Agreement and Waiver:

# The Terms, together with the 'Privacy Policy' and 'Guidelines and Policies', shall constitute the entire agreement between you and us concerning the Services. No failure or delay by us in exercising any right, power or privilege under the Terms shall operate as a waiver of such right or acceptance of any variation of the Terms and nor shall any single or partial exercise by either party of any right, power or privilege preclude any further exercise of that right or the exercise of any other right, power or privilege.

# Severability:

# If any provision of these Terms is deemed unlawful, invalid, or unenforceable by a judicial court for any reason, then that provision shall be deemed severed from these Terms, and the remainder of the Terms shall continue in full force and effect.

# Partnership or Agency:

# None of the provisions of these Terms shall be deemed to constitute a partnership or agency between you and Zomato and you shall have no authority to bind Zomato in any form or manner, whatsoever.

# Governing Law/Waiver:

# (a) For Customers residing in India: These Terms shall be governed by the laws of India. The Courts of New Delhi shall have exclusive jurisdiction over any dispute arising under these terms.

# (b) For Customers residing in UAE: These Terms shall be governed by the laws of UAE. The Courts of Dubai shall have exclusive jurisdiction over any dispute arising under these terms.

# (c) For Customers residing in Lebanon: These Terms shall be governed by the laws of Lebanon. The Courts of Beirut shall have exclusive jurisdiction over any dispute arising under these terms.

# (d) For Customers residing in the United States: These Terms shall be governed in all respects by the laws of the State of Washington as they apply to agreements entered into and to be performed entirely within the State of Washington between Washington residents, without regard to conflict of law provisions. You agree that any claim or dispute you may have against Zomato must be resolved exclusively by a state or federal court located in Seattle, Washington. You agree to submit to the personal jurisdiction of the courts located within Seattle, Washington for the purpose of litigating all Claims that arise between You and Zomato.

# (e) For all Customers: YOU MUST COMMENCE ANY LEGAL ACTION AGAINST US WITHIN ONE (1) YEAR AFTER THE ALLEGED HARM INITIALLY OCCURS. FAILURE TO COMMENCE THE ACTION WITHIN THAT PERIOD SHALL FOREVER BAR ANY CLAIMS OR CAUSES OF ACTION REGARDING THE SAME FACTS OR OCCURRENCE, NOTWITHSTANDING ANY STATUTE OF LIMITATIONS OR OTHER LAW TO THE CONTRARY. WITHIN THIS PERIOD, ANY FAILURE BY US TO ENFORCE OR EXERCISE ANY PROVISION OF THESE TERMS OR ANY RELATED RIGHT SHALL NOT CONSTITUTE A WAIVER OF THAT RIGHT OR PROVISION.

# Carrier Rates may Apply:

# By accessing the Services through a mobile or other device, you may be subject to charges by your Internet or mobile service provider, so check with them first if you are not sure, as you will be solely responsible for any such costs incurred.

# Linking and Framing:

# You may not frame the Services. You may link to the Services, provided that you acknowledge and agree that you will not link the Services to any website containing any inappropriate, profane, defamatory, infringing, obscene, indecent, or unlawful topic, name, material, or information or that violates any intellectual property, proprietary, privacy, or publicity rights. Any violation of this provision may, in our sole discretion, result in termination of your use of and access to the Services effective immediately.

# XVII. Notice of copyright infringement
# Zomato shall not be liable for any infringement of copyright arising out of materials posted on or transmitted through the Zomato Platform, or items advertised on the Zomato Platform, by end users or any other third parties. We respect the intellectual property rights of others and require those that use the Services to do the same. We may, in appropriate circumstances and at our discretion, remove or disable access to material on the Services that infringes upon the copyright rights of others. We also may, in our discretion, remove or disable links or references to an online location that contains infringing material or infringing activity. In the event that any Customers of the Services repeatedly infringe on others' copyrights, we may in our sole discretion terminate those individuals' rights to use the Services If you believe that your copyright has been or is being infringed upon by material found in the Services, you are required to follow the below procedure to file a notification:

# i. Identify in writing the copyrighted material that you claim has been infringed upon;

# ii. Identify in writing the material on the Services that you allege is infringing upon copyrighted material, and provide sufficient information that reasonably identifies the location of the alleged infringing material (for example, the user name of the alleged infringer and the business listing it is posted under);

# iii. Include the following statement: "I have a good faith belief that the use of the content on the Services as described above is not authorized by the copyright owner, its agent, or law";

# iv. Include the following statement: "I swear under penalty of perjury that the information in my notice is accurate and I am the copyright owner or I am authorized to act on the copyright owner's behalf";

# v. Provide your contact information including your address, telephone number, and e-mail address (if available);

# vi. Provide your physical or electronic signature;

# vii. Send us a written communication to legal@zomato.com

# You may be subject to liability if you knowingly make any misrepresentations on a take-down notice.'''

# sentences = summarize(source)
# l = joblib.load('Text_SVM.pkl')
# count_vect = l[1]
# transformer = l[2]
# model = l[0]
# d = {}
# for sentence in sentences:
#     mc = count_vect.transform([sentence])
#     m = transformer.transform(mc)
#     output = model.predict(m)
#     if output[0] not in d.keys():
#         newlist = list()
#         newlist.append(sentence)
#         d[output[0]] = newlist
#     else:
#         d[output[0]].append(sentence)
# print(d)
