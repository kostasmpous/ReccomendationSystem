import random
import pandas as pd
import math
import heapq
from sklearn.metrics.pairwise import cosine_similarity

#file initialize
with open('config.txt') as file:
    line = file.readlines()
for l in line:
    N, M, K, R, T = l.split(" ")
    N,M,K,R,T = int(N),int(M),int(K),int(R),int(T)

#initialize array
Matrix = [[0 for x in range(M)] for y in range(N)]#contains the true data about movies and users
for x in range(N):
    for y in range(M):
        Matrix[x][y] = round(random.uniform(1, 5),2)

#Transform list to Dataframe array
movies = []#list of all movies
for mov in range(M):
    movies.append("Movie "+ str(mov))

users = []#list of all users
for user in range(N):
    users.append(user)

data = pd.DataFrame(Matrix,columns=movies)

#create 2 empty Dataframe arrays (NaN) for training(1.Pearson 2.Cos.simil)
training_data1 = pd.DataFrame(index=users, columns=movies)
training_data2 = pd.DataFrame(index=users, columns=movies)

#calculate number of elements im gonna use for training
total_items = N*M
use_items = math.ceil((total_items * T)/100)
itemsUser = math.floor(use_items/N)

#fill the array with T% items of original data
oldNumbers=[]
movie = 0
for x in range(N):
    oldNumbers.clear()
    for y in range(itemsUser):
        while True:
            movie = random.randint(0,M-1)
            if movie not in oldNumbers:
                break
        oldNumbers.append(movie)
        training_data1.at[x, "Movie "+ str(movie)] = data.xs(x)["Movie "+ str(movie)]

#calculate Pearson for training_data1
Pearson_data = training_data1.astype(float).T.corr(method='pearson')
for x in range(N):
    for y in range(M):
        if(x==y):
            Pearson_data[x][y] = -1

#calculate Cosine Sim. for training data 2
training_data2 = training_data1.copy()
training_data2.fillna(0,inplace=True)
Cosine_data = cosine_similarity(training_data2.astype(float).T,None,True)
for x in range(N):
    for y in range(M):
        if(x==y):
            Cosine_data[x][y] = -1

Cos_data = pd.DataFrame(Cosine_data,index=movies,columns=movies)#cosine similarity data with rows and collumns movies

#-------------------------------Calculate K-neighboor for training_data1(Pearson)--------------------------------------

final = pd.DataFrame(index=users,columns=[0,1,2,3])#create the final table for Pearson

#find k-neighboors and add them to a table(final)
listofusers=[]
for x in range(M):
    for y in range(N):
        listofusers.append(Pearson_data[x][y])
    list2=[]
    list2 = heapq.nlargest(K,zip(listofusers,users))#return the k highest neighboors with the pearson correlation in tuple
    listUsers = []
    listWeights = []
    for i in list2:#im using 2 lists to seperate weights and users and i will add them later to different table (final)
        listUsers.append(i[1])
        listWeights.append(round(i[0], 2))
    final.at[x, 0] = listUsers#add the list with the users in the final array
    final.at[x,1] = listWeights#add the list with the weights
    listofusers.clear()

#find which movies every person havent seen
listofmovies = []
for x in range(M):
    for y in range(N):
        if math.isnan(training_data1.at[x, "Movie "+ str(y)]):
            listofmovies.append((y,x))#add to tuple the movie and the user (movie,user)

#add the movies to the same table
for u in range(N):
    listunseen = []
    for m in listofmovies:
        if(u == m[1]):
            listunseen.append(m[0])
    final.at[u,2] = listunseen


for x in range(M):#For every user
    movies2 = final.at[x,2]#The movies he havent seen
    neighboors = final.at[x,0]#His neighboors
    weights = final.at[x,1]#weights of movies
    listSums=[]#list with all the sums of the movies
    limit = 0

    for mv in movies2:
        limit = limit+1
        if limit == R-1:
            break
        divisor = 0
        sum =0
        for nb in neighboors:
            if math.isnan(training_data1.at[nb,"Movie " + str(mv)]):#check if NaN
               divisor = divisor
            else:
                sum = sum + (training_data1.at[nb,"Movie " + str(mv)])*final.at[x,1][final.at[x,2].index(mv)]
                divisor = divisor + final.at[x,1][final.at[x,2].index(mv)]
        if (divisor == 0):
            divisor = 1
        listSums.append(round(sum/divisor,2))
    final.at[x,3] = listSums#add summs to list and to final table

#now adding the predictions in table of predictions

table_prediction1 = pd.DataFrame(index=users,columns=movies)

for x in range(M):
    listmov = final.at[x,2]#list of movies the user didnt watch
    listPred = final.at[x,3]#list of predictions for this user
    index = 0
    for m in listmov:
        if (index<=len(listPred)-1):
            table_prediction1.at[x,"Movie "+str(m)] = listPred[index]#add prediction in table prediction
            index = index + 1
dicofusers1={}

for x in range(M):
    listofpredictions = []
    listofmovies =[]

    for y in range(N):
        if not math.isnan(table_prediction1.at[x,"Movie "+str(y)]):
            listofpredictions.append((table_prediction1.at[x,"Movie "+str(y)]))
            listofmovies.append("Movie "+str(y))

    listfinal = []  # contains the highests predictions for this user
    listfinal = heapq.nlargest(R, zip(listofpredictions, listofmovies))#find the R highests movies for user x
    dicofusers1[x]=listfinal#dictionary of    user:[(prediction,'Movie x'),(prediction,'Movie x')]

#----------------------------------------Calculate cosine similarity predictions--------------------------------------

final2 = pd.DataFrame(index=movies,columns=[0,1,2,3])

#add the similar movies to the final array
listofmovies=[]
for x in range(N):
    for y in range(N):
        listofmovies.append(Cosine_data[x][y])
    list2=[]
    list2 = heapq.nlargest(K,zip(listofmovies,movies))#return the 3 highest neighboors with the cosine similarity in tuple
    list3 =[]
    list4=[]
    for i in list2:#im using 2 lists to remove the cosine sim. from the previous list because i only need the movies
        list3.append(i[1])
        list4.append(round(i[0],2))
    final2.at["Movie " + str(x), 0] = list3#add the list with the movies in the final array
    final2.at["Movie " + str(x), 1] = list4#add the list with the weights
    listofmovies.clear()

#find which movies every person havent seen
listofmovies2 = []

for x in range(M):
    for y in range(N):
        if (training_data2.at[x, "Movie "+ str(y)] == 0):
            listofmovies2.append((y,x))

#add the users that havent seen this movie
for u in range(N):
    listunseen1=[]
    for m in listofmovies2:
        if(u==m[1]):
            listunseen1.append(m[0])
    final2.at["Movie " + str(u),2] = listunseen1


#make the predictions
for y in range(M):
    listuserun = final2.at["Movie " + str(y),2]
    listofmov = final2.at["Movie " + str(y),0]
    weights = final2.at["Movie " + str(y),1]
    listsums2=[]
    for us in listuserun:
        sum = 0
        divisor = 0
        for mv in listofmov:
            if (training_data2.at[us,mv] == 0):
                divisor = divisor
            else:
                sum = sum +(training_data2.at[us,mv])*final2.at["Movie "+str(y),1][final2.at["Movie "+str(y),0].index(mv)]
                divisor = divisor + final2.at["Movie "+str(y),1][final2.at["Movie "+str(y),0].index(mv)]
        if (divisor == 0):
            divisor = 1

        listsums2.append(round(sum/divisor,2))
    final2.at["Movie "+str(y),3] = listsums2#add prediction to final2
table_prediction2 = pd.DataFrame(index = users,columns=movies)

#fill the prediction table with the predictions
for y in range(M):
    listofuser = final2.at["Movie " + str(y),2]
    listofpred = final2.at["Movie " + str(y),3]
    index=0
    for us in listofuser:
        table_prediction2.at[us,"Movie "+str(y)]= listofpred[index]
        index = index+1

#find k higher prediction to propose
dicofusers2={}
for x in range(M):
    listofpredictions = []
    listofmovies =[]

    for y in range(N):
        if not math.isnan(table_prediction2.at[x,"Movie "+str(y)]):
            listofpredictions.append((table_prediction2.at[x,"Movie "+str(y)]))
            listofmovies.append("Movie "+str(y))

    listfinal = []  # contains the highests predictions for this user
    listfinal = heapq.nlargest(R, zip(listofpredictions, listofmovies))#find the R highests movies for user x
    dicofusers2[x]=listfinal#dictionary of    user:[(prediction,'Movie x'),(prediction,'Movie x')]

#-----------------------------------------------MAP MEASURE-----------------------------------------------------------
def MAP(dicproposals):
    numenator = 0
    denominator = 0
    sum=0
    divider = 0
    for key, value in dicproposals.items():
        for tuple in value:
            if(data.at[key,tuple[1]]>3.5):
                numenator = numenator+1
                denominator = denominator +1
                sum=sum+(numenator/denominator)
                divider = divider +1
            else:
                denominator = denominator+1
    final = round(sum/divider,2)
    with open('results.txt', 'a') as file:
        file.write("For M,N,K,R,T = " + str(N) + ", " + str(M) + ", " + str(K) + ", " + str(R) + ", " + str(T) +'\n')
        file.write("MAP = " + str(final) + '\n')
#-----------------------------------------------F-measure-------------------------------------------------------------
def F_meas(dicproposals):
    TP=0
    FP=0
    #calculate TP and FP
    for key, value in dicproposals.items():
        for tuple in value:
            if(data.at[key,tuple[1]]>3.5):#check if the movie i proposed is indeed above 3.5
                TP=TP+1
            else:
                FP=FP+1
    #calculate FN
    FN=0
    for x in range(M):
        for y in range(N):
            truescore = data.at[x,"Movie " + str(y)]
            movie = "Movie " + str(y)
            find = 0
            if(truescore>3.5):#if the truescore is above 3.5 check if i proposed it
                for key, value in dicproposals.items():
                    for tuple in value:
                        if (key == x and tuple[1] == movie):#if key = user and movie = with movie inside the tuple
                            find =1
                if(find==0):
                    FN = FN+1

    P = TP/(TP + FP)
    R = TP/(TP + FN)
    F = (2*P*R)/(P+R)
    F = round(F,2)
    with open('results.txt', 'a') as file:
        file.write("For M,N,K,R,T = " + str(N) + ", " + str(M) + ", " + str(K) + ", " + str(R) + ", " + str(T) + '\n')
        file.write("F-measure = " + str(F) + '\n')

#call the functions
MAP(dicofusers1)
MAP(dicofusers2)
F_meas(dicofusers1)
F_meas(dicofusers2)