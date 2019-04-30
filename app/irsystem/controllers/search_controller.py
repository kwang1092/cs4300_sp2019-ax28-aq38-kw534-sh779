from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import numpy as np
import csv
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from nltk.tokenize import TreebankWordTokenizer
from sklearn import linear_model
import Levenshtein

apple_mult = 1.24
project_name = "sellPhones"
net_id = "Alvin Qu: aq38, Andrew Xu: ax28, Kevin Wang: kw534, Samuel Han: sh779"

@irsystem.route('/', methods=['GET'])
def search():
    check = False
    mate = False
    check2 = False
    mate2 = False
    past = request.args.get('past')
    past2 = request.args.get('past2')
    past3 = request.args.get('past3')
    past4 = request.args.get('past4')
    if past:
        check = True
        mate=True
    else:
        mate = True

    if past2 == "True":
        flag = True
    else:
        flag = False

    if past3:
        check2 = True
        mate2=True
    else:
        mate2 = True

    if past4 == "True":
        flag2 = True
    else:
        flag2 = False

    input_arr = []
    final = [[0],[0]]

    condition = request.args.get('condition')
    budget = request.args.get('budgets')
    feature_list = request.args.getlist('feature')
    old_phone = ""
    old_phone = request.args.get('old_phone')
    feature_text = request.args.get('feature_text')

    if not feature_text:
        feature_text = ""

    if not old_phone:
        old_phone = ""




    if not feature_list :
        return render_template('search.html', name=project_name,netid=net_id, check=check, check2=check2, mate2=mate2,  mate=mate, flag=flag, flag2 = flag2,
                                condition=condition, names=[], urls = [], budget=str(budget), features = [], close_words=[],scores=[],price=[])

    if budget and feature_list and condition:

        def main(budget, feature_list,condition):
            phones = {}
            labels = []
            with open('app/static/gsmphones.csv', mode='r') as csv_file:
                csv_reader = csv.reader(csv_file)
                for i,row in enumerate(csv_reader):
                    if i == 0:
                        labels = row
                    else:
                        phones[row[1]] = row[2:]

            phones.pop("Samsung Galaxy S10 5G")

            for elt in phones:
                l = len(phones[elt])
                if l<100:
                    while(len(phones[elt])<100):
                        phones[elt].append(0)

            label_to_index = {labels[i].lower(): i-2 for i in range(len(labels))}
            label_to_index["ppi"] = 81
            label_to_index["storage"] = 82
            label_to_index["ram"] = 83
            label_to_index["rear camera"] = 85
            label_to_index["front camera"] = 86
            label_to_index["age"] = 87
            label_to_index["face"]= 88
            label_to_index["finger"] = 89

            def getResolution(res):
                lst = []
                if len(res)>=12:
                    lst = re.findall(r"\d+\.?\d?",res)
                if len(lst)>1:
                    return (int(lst[0]),int(lst[1]))
                else:
                    return (0,0)

            def getDiagonalSize(size):
                if len(size)>=10:
                    return float(re.findall(r"\d+\.?\d?",size)[0])

            def getThickness(dim):
                lst = re.findall(r"\d+\.?\d+",dim)
                if len(lst)>2:
                    return(float(lst[2]))
                else:
                    return 0

            new_phones = {}
            for i,phone in enumerate(phones):
                a,b = getResolution(phones[phone][label_to_index["resolution"]])
                size= getDiagonalSize(phones[phone][label_to_index["size"]])
                phones[phone][label_to_index["size"]] = size
                diag= np.sqrt(a**2+b**2)
                if size!=None and size>1.7 and size<=7.0 and a!=0 and b!=0:
                    phones[phone][label_to_index["ppi"]] = diag/size
                    card_slot = phones[phone][label_to_index["card slot"]]
                    if card_slot.lower()=="no":
                        phones[phone][label_to_index["card slot"]] = 0
                    else:
                        phones[phone][label_to_index["card slot"]] = 1
                    new_phones[phone] = phones[phone]

            phones = new_phones

            new_phones = {}
            for i,phone in enumerate(phones):
                dim = phones[phone][label_to_index["dimensions"]]
                if len(dim) >= 12 and "mm" in dim:
                    phones[phone][label_to_index["dimensions"]] = getThickness(dim)
                    new_phones[phone] = phones[phone]

            phones = new_phones

            def convertGB(num, byte):
                if byte.lower()=="mb":
                    return float(num)/1024
                else:
                    return num

            new_phones = {}
            for i,phone in enumerate(phones):
                lst = re.findall(r"[\d.]+",phones[phone][label_to_index["internal"]])
                byte= re.findall(r"[a-zA-Z]+",phones[phone][label_to_index["internal"]])
                if len(lst)>1:
                    mem = 0
                    ram = 0
                    if "ROM" in byte and "RAM" in byte:
                        ram = convertGB(lst[-2],byte[byte.index('RAM')-1])
                        mem = convertGB(lst[-1],byte[byte.index('ROM')-1])
                    else:
                        try:
                            ram = convertGB(lst[-1],byte[byte.index('RAM')-1])
                        except:
                            ram = convertGB(lst[-1],byte[-1])
                        if float(lst[-1]) >= 100:
                            lst.pop(len(lst)-1)
                        lst = [float(elt) for elt in lst]
                        mem = convertGB(sorted(lst)[-1],byte[0])
                    phones[phone][label_to_index["storage"]] = float(mem)
                    phones[phone][label_to_index["ram"]] = float(ram)
                    new_phones[phone] = phones[phone]
                elif len(lst)==1 and float(lst[0]) > 0:
                    phones[phone][label_to_index["storage"]] = float(convertGB(lst[-1],byte[0]))
                    phones[phone][label_to_index["ram"]] = 0
                new_phones[phone] = phones[phone]

            phones=new_phones

            new_phones = {}
            for i,phone in enumerate(phones):
                temp = phones[phone][label_to_index["price"]]
                price = re.findall(r"\d+\.?\d?",temp)
                curr = re.findall(r"[a-zA-Z]+",temp)
                if len(temp) > 0:
                    if curr[-1].lower()=="eur":
                        phones[phone][label_to_index["price"]] = round(float(price[0]) * 1.13, 2)
                    else:
                        phones[phone][label_to_index["price"]] = round(float(price[0]) * 0.014, 2)
                    new_phones[phone] = phones[phone]
            phones = new_phones

            def getBattery(dim):
                if type(dim)!=str:
                    return 0.0
                lst = re.findall(r"\d+\.?\d?",dim)
                if len(lst)>0:
                    return float(lst[0])
                else:
                    return 0.0

            new_phones = {}
            for i,phone in enumerate(phones):
                if "face" in (phones[phone][label_to_index["sensors"]]).lower():
                    phones[phone][label_to_index["face"]] = 1.0
                else:
                    phones[phone][label_to_index["face"]] = 0.0
                if "finger" in (phones[phone][label_to_index["sensors"]]).lower():
                    phones[phone][label_to_index["finger"]] = 1.0
                else:
                    phones[phone][label_to_index["finger"]] = 0.0
                new_phones[phone] = phones[phone]

                b = getBattery(phones[phone][label_to_index["battery"]])
                phones[phone][label_to_index["battery"]] = b
                new_phones[phone] = phones[phone]

                water = phones[phone][label_to_index["waterproof"]]
                if type(water)==str and "yes" in water.lower():
                    phones[phone][label_to_index["waterproof"]] = 1.0
                else:
                    phones[phone][label_to_index["waterproof"]] = 0.0

                sim = phones[phone][label_to_index["sim"]].lower()
                if "dual" in sim:
                    phones[phone][label_to_index["sim"]] = 1
                else:
                    phones[phone][label_to_index["sim"]] = 0

                temp = phones[phone][label_to_index["video"]]
                fps  = re.findall(r"\d+[a-zA-z][a-zA-z]+",temp)
                if len(fps)==0:
                    fps = 0
                else:
                    fps = float(re.findall(r"\d+",fps[0])[0])
                temp = re.findall(r"\d+\.?\d?",temp)
                if len(temp) > 0:
                    phones[phone][label_to_index["video"]] = float(temp[0])*(fps/30)
                else:
                    phones[phone][label_to_index["video"]] = 240.0

                temp = phones[phone][label_to_index["announced"]]
                temp = re.findall(r"\d+\.?\d+",temp)
                if len(temp)>0:
                    phones[phone][label_to_index["age"]] = 2019-float(temp[0])
                    if phones[phone][label_to_index["age"]] <= 1:
                        phones[phone][label_to_index["announced"]] = 1.
                    else:
                        phones[phone][label_to_index["announced"]] = 0.
                else:
                    phones[phone][label_to_index["age"]] = 7
                    phones[phone][label_to_index["announced"]] = 0.

                if "yes" in phones[phone][label_to_index["3.5mm jack"]].lower():
                    phones[phone][label_to_index["3.5mm jack"]] = 1.0
                else:
                    phones[phone][label_to_index["3.5mm jack"]] = 0.0

                temp = phones[phone][label_to_index["cpu"]].lower()
                if "octa" in temp:
                    phones[phone][label_to_index["cpu"]] = 8.0
                elif "hexa" in temp:
                    phones[phone][label_to_index["cpu"]] = 6.0
                elif "quad" in temp:
                    phones[phone][label_to_index["cpu"]] = 4.0
                elif "dual" in temp:
                    phones[phone][label_to_index["cpu"]] = 2.0
                else:
                    phones[phone][label_to_index["cpu"]] = 1.0

                dual = phones[phone][label_to_index["dual"]]
                front= re.findall(r"\d+",phones[phone][label_to_index["single"]])
                if len(front)>1 and ("mp" in phones[phone][label_to_index["single_1"]].lower() or
                                     "mp" in str(phones[phone][label_to_index["dual"]]).lower()):
                    cam1 = float(front[0])
                    if dual!="" and phone.split(" ")[0] != 'Acer':
                        rear = re.findall(r"\d+",phones[phone][label_to_index["dual"]])
                        phones[phone][label_to_index["dual"]] = 1
                        cam2 = float(rear[0])
                        phones[phone][label_to_index["front camera"]]= min(cam1,cam2)
                        phones[phone][label_to_index["rear camera"]] = max(cam1,cam2)
                    else:
                        rear = re.findall(r"\d+",phones[phone][label_to_index["single_1"]])
                        phones[phone][label_to_index["dual"]] = 0
                        cam2 = float(rear[0])
                        if len(rear)>0:
                            phones[phone][label_to_index["front camera"]]= min(cam1,cam2)
                            phones[phone][label_to_index["rear camera"]] = max(cam1,cam2)
                else:
                    phones[phone][label_to_index["front camera"]] = 0
                    phones[phone][label_to_index["dual"]] = 0
                    if len(front) > 0 and "mp" in phones[phone][label_to_index["single"]].lower():
                        phones[phone][label_to_index["rear camera"]] = float(front[0])
                    else:
                        phones[phone][label_to_index["rear camera"]] = 0

                new_phones[phone] = phones[phone]

            label_to_index["thickness"] = label_to_index["dimensions"]
            label_to_index.pop("dimensions")

            phones = new_phones

            rel_labels = [label_to_index["model image"],label_to_index["announced"],
                          label_to_index["cpu"],label_to_index["ppi"],label_to_index["storage"],
                          label_to_index["video"],label_to_index["dual"],label_to_index["ram"],
                          label_to_index["thickness"],label_to_index["sim"],label_to_index["card slot"],
                          label_to_index["rear camera"],label_to_index["front camera"],
                          label_to_index["size"],label_to_index["3.5mm jack"],label_to_index["waterproof"],
                          label_to_index["battery"],label_to_index["face"],label_to_index["finger"],
                          label_to_index["price"]]

            features = {}
            for i,phone in enumerate(phones):
                features[phone] = []
                for label in rel_labels:
                    features[phone].append(phones[phone][label])
                #features[phone] = np.array(features[phone])

            feat_to_index = {"announced":0,"cpu":1,"ppi":2,"storage":3,
                             "video":4,"dual":5,"ram":6,"thickness":7,"sim":8,
                             "card slot":9,"rear camera":10,"front camera":11,
                             "size":12,"3.5mm jack":13,"waterproof":14,
                             "battery":15,"face":16,"finger":17,"price":18}

            feature_mat = np.zeros((len(phones),len(features["Apple iPhone XS Max"])-1))

            phone_to_index = {}
            index_to_phone = []
            for i,phone in enumerate(features):
                feature_mat[i] = features[phone][1:]
                feature_mat[i,feat_to_index["thickness"]] = 20-feature_mat[i,feat_to_index["thickness"]]
                phone_to_index[phone] = i
                index_to_phone.append(phone)
            prices = feature_mat[:,feat_to_index["price"]]

            for i in range(len(feature_mat[0])):
                if i!=feat_to_index["price"]:
                    feature_mat[:,i] = feature_mat[:,i]/max(feature_mat[:,i])

            new_phones = {}
            new_prices = {}
            new_index  = {}
            old_phones = {}
            old_index  = {}
            for i in range(len(feature_mat)):
                if True not in np.isnan(feature_mat[i,:]):
                    if feature_mat[i,0]==1:
                        new_phones[index_to_phone[i]] = i
                        new_index[index_to_phone[i]]  = len(new_index)
                        new_prices[index_to_phone[i]] = feature_mat[i,feat_to_index["price"]]
                    else:
                        old_phones[index_to_phone[i]] = feature_mat[i,:]
                        old_index[len(old_index)]     = index_to_phone[i]

            phones_tr = []
            prices_tr = []
            new_index = {}
            for phone in new_phones:
                phone_vec = feature_mat[phone_to_index[phone],:]
                if True not in np.isnan(phone_vec):
                    new_index[len(phones_tr)] = phone
                    phones_tr.append(phone_vec[1:14])
                    prices_tr.append(new_prices[phone])
            phones_tr = np.array(phones_tr)
            prices_tr = np.array(prices_tr)

            phones_te = []
            for phone in old_phones:
                phone_vec = feature_mat[phone_to_index[phone],:]
                if True not in np.isnan(phone_vec):
                    phones_te.append(phone_vec[1:14])
            phones_te = np.array(phones_te)

            poly = PolynomialFeatures(degree=2)
            X_tr = poly.fit_transform(phones_tr)
            X_te = poly.fit_transform(phones_te)

            poly_model = linear_model.LinearRegression()
            poly_model.fit(X_tr, prices_tr)
            poly_pred  = poly_model.predict(X_te)

            new_ptoi = {}
            for phone in new_index:
                new_ptoi[new_index[phone]] = phone

            old_ptoi = {}
            for phone in old_index:
                old_ptoi[old_index[phone]] = phone

            def priceDiff(p,curr):
                return (phones[p][label_to_index["price"]]-curr)/phones[p][label_to_index["price"]]

            #NEW
            for p in old_ptoi:
                curr_price = feature_mat[phone_to_index[p]][feat_to_index["price"]]
                age = phones[p][label_to_index["age"]]

                curr_price *= 0.75
                curr_price -= 0.05*(1-max(0.5,priceDiff(p,poly_pred[old_ptoi[p]])))*age*curr_price
                feature_mat[phone_to_index[p]][feat_to_index["price"]] = curr_price


            #loading preprocessed review dictionaries
            with open('app/static/concat_reviews.json', 'r') as fp:
                concat_reviews = json.load(fp)

            with open('app/static/review_stuff.json', 'r') as fp:
                review_stuff = json.load(fp)

            with open('app/static/sent_anal_dict.json', 'r') as fp:
                sent_anal_dict = json.load(fp)

            with open('app/static/ratings.json', 'r') as fp:
                ratings = json.load(fp)

            #creating vocab from reviews and list of phones that we have reviews for
            review_vocab = review_stuff['vocab']
            review_phonenames = review_stuff['phonenames']
            n_phones = len(review_phonenames)
            n_vocab  = len(review_vocab)

            #function for building inverted index
            def build_inv_idx(lst):
                """ Builds an inverted index.

                Params: {lst: List}
                Returns: Dict (an inverted index of phones)
                """
                inverted_idx = {}
                for idx in range(0,len(lst)):
                    inverted_idx[lst[idx]] = idx
                return inverted_idx

            #Create matrix from saves json of look-around sentiment analysis polarity values
            lookaround_matrix = np.zeros((n_phones,n_vocab))
            for key,value in sent_anal_dict.items():
                coord = key.split(",")
                x,y = int(coord[0]),int(coord[1])
                lookaround_matrix[x,y] = value

            #build inverted indexes for vocab and reviewed phone names
            review_vocab_invidx = build_inv_idx(review_vocab)
            review_names_invidx = build_inv_idx(review_phonenames)

            review_list = [concat_reviews[p] for p in concat_reviews]
            vectorizer = TfidfVectorizer(stop_words = 'english',encoding='utf-8',lowercase=True)
            my_matrix = vectorizer.fit_transform(review_list).transpose()
            u, s, v_trans = svds(my_matrix, k=100)
            words_compressed, _, docs_compressed = svds(my_matrix, k=30)
            docs_compressed = docs_compressed.transpose()
            word_to_index = vectorizer.vocabulary_
            index_to_word = {i:t for t,i in word_to_index.items()}
            words_compressed = normalize(words_compressed, axis = 1)

            def closest_words(word_in, k = 10):
                if word_in not in word_to_index: return "Not in vocab."
                sims = words_compressed.dot(words_compressed[word_to_index[word_in],:])
                asort = np.argsort(-sims)[:k+1]
                return [(index_to_word[i],sims[i]/sims[asort[0]]) for i in asort[1:]]

            def query_word(word):
                close_words = closest_words(word)
                return [word,close_words[0][0],close_words[1][0]]

            #Taking input from SVD
            words_from_svd = []
            for word in feature_text.split(" "):
                words_from_svd += query_word(word)
            n_words = len(words_from_svd)
            n_phones = len(review_phonenames)
            query_matrix = np.zeros((n_phones,n_words))

            new_string = []
            for word in words_from_svd:
                if word in review_vocab:
                    new_string.append(word)

            #RANKINGS using custom input
            for phone in review_phonenames:
                p = review_names_invidx[phone]
                for i,word in enumerate(new_string):
                    w = review_vocab_invidx[word]
                    if i%3 != 0:
                        query_matrix[p,i] = lookaround_matrix[p,w] / 50
                    else:
                        query_matrix[p,i] = lookaround_matrix[p,w]

            #Outputting ranking based on social component
            query_matrix = np.sum(query_matrix, axis=1)
            query_matrix = query_matrix / len(new_string)

            #WITH RATINGS (to be merged with cell above)
            for phone in review_phonenames:
                p = review_names_invidx[phone]
                rating = ratings[phone]
                rating_effect = 1.0
                ratio = rating/5.0
                polarity = query_matrix[p]

                if rating >= 4 and polarity > 0:
                    rating_effect = 1.3*ratio
                elif rating >= 4 and polarity < 0:
                    rating_effect = -1.3*ratio
                elif rating <= 2.5 and polarity > 0:
                    rating_effect = -1.0*ratio
                elif rating <= 2.5 and polarity < 0:
                    rating_effect = 1.3+(1-ratio)

                query_matrix[p] = rating_effect*polarity

            ranking_asc = list(np.argsort(query_matrix))
            ranking_desc = ranking_asc[::-1]

            phone_to_review = {}
            for i in range(n_phones):
                phone_to_review[review_phonenames[ranking_desc[i]]] = query_matrix[ranking_desc[i]] / query_matrix[ranking_desc[0]]
            for phone in phones:
                if phone not in phone_to_review:
                    phone_to_review[phone] = 0

            bin_feats = {"3.5mm jack":0,"dual":1,"sim":2,"announced":3,"card slot":4,"waterproof":5,"face":6,"finger":7}

                    #Need user input
                    # min_price = 800
                    # max_price = 1600
                    # price_range = budget

            prices = feature_mat[:,feat_to_index["price"]]
            budget = str(budget)
            starting = int(0)
            ending = int(budget)
            price_range = np.intersect1d(np.where(prices>=starting)[0],np.where(prices<ending)[0])

            if condition == "new":
                condition = 1
            else:
                condition = 0
            query_feat = feature_list

            #query_feat = ["ram","front camera","cpu","rear camera"]
            # price_range= luxury

            if old_phone:
                old_query = old_phone
            else:
                old_query = ""


            def checkBinary(idx,query_feat):
                phone = index_to_phone[idx]
                for feat in query_feat:
                    if feat in bin_feats and feature_mat[idx][feat_to_index[feat]]!=1:
                        return False
                return True

            def getRank(dic):
                ranks = []
                for elt in dic:
                    ranks.append((elt,dic[elt]))
                ranks = sorted(ranks,key=lambda x: x[1])
                for elt in ranks:
                    print(elt)

            def edit_distance_search(query, names):
                result = []
                for name in names:
                    score = Levenshtein.distance(query.lower(), name.lower())
                    result.append((score,name))
                result = sorted(result, key=lambda x: x[0])
                return result

            if len(query_feat)==0:
                query_feat = set(feat_to_index.keys())-set(bin_feats.keys())

            results = {}
            ranked_results = []
            best_match = edit_distance_search(old_query,phones.keys())[0][1]
            best_dist  = edit_distance_search(old_query,phones.keys())[0][0]
            best_match_vec = feature_mat[phone_to_index[best_match]]
            best_vec = np.zeros(len(query_feat))

            physical_feat = True
            for elt in query_feat:
                if elt not in ["size","thickness"]:
                    physical_feat = False

            query_vec = np.zeros(len(query_feat))
            for i,feat in enumerate(query_feat):
                best_vec[i]  = best_match_vec[feat_to_index[feat]]
                query_vec[i] = max(feature_mat[:,feat_to_index[feat]])

            cossim = {}
            for p in price_range:
                if feature_mat[p][0] == condition and index_to_phone[p] != old_query and checkBinary(p,query_feat) \
                    and feature_mat[p][feat_to_index["thickness"]]>0.2 and feature_mat[p][feat_to_index["thickness"]]<1:
                    temp = np.zeros(len(query_feat))
                    for i,feat in enumerate(query_feat):
                        temp[i] = feature_mat[p,feat_to_index[feat]]
                    if checkBinary(phone_to_index[best_match],query_feat):
                        cossim[p] = np.dot(temp,best_vec)/np.linalg.norm(temp)*np.linalg.norm(best_vec)
                    else:
                        cossim[p] = np.dot(temp,temp)/np.linalg.norm(temp)**2
            cossim_lst = []
            for idx in cossim:
                cossim_lst.append([idx,cossim[idx]])
            cossim_lst = np.array(cossim_lst)
            cossim_lst[:,1] /= max(cossim_lst[:,1])
            sim_index = np.argsort(cossim_lst[:,1])[::-1]
            cossim_min = min(cossim_lst[:,1])
            for idx in sim_index:
                results[index_to_phone[int(cossim_lst[idx][0])]] = (cossim_lst[idx,1]-cossim_min)/(max(cossim_lst[:,1])-cossim_min+1e-5)

            query_mat = []
            prange_to_phone = []
            for p in price_range:
                if feature_mat[p][0] == condition and index_to_phone[p] != old_query and checkBinary(p,query_feat) \
                    and feature_mat[p][feat_to_index["thickness"]]>0.2 and feature_mat[p][label_to_index["thickness"]]!=1:
                    prange_to_phone.append(index_to_phone[p])
                    query_mat.append(feature_mat[p][[feat_to_index[feat] for feat in query_feat]])
            query_mat = np.array(query_mat)

            cossim_vec = np.zeros(len(phones))
            for i,vec in enumerate(query_mat):
                vec /= query_vec
                brand = prange_to_phone[i].split(" ")[0]
                if not physical_feat:
                    if brand == "Apple" or brand == "Google":
                        vec *= 1.2
                    elif brand == "Samsung":
                        vec *= 1.05
                cossim_vec[i] = np.linalg.norm(vec)
                cossim_vec[i] *= 1+0.125*phone_to_review[prange_to_phone[i]]

            rankings = np.where(cossim_vec>0)
            rankings = np.argsort(cossim_vec[rankings])[::-1]
            cossim_vec /= max(cossim_vec)
            for idx in rankings:
                if best_dist > 5:
                    results[prange_to_phone[idx]] += cossim_vec[idx]
                else:
                    results[prange_to_phone[idx]] += 3*cossim_vec[idx]
                    results[prange_to_phone[idx]] /= 4

            final_rank = []
            for phone in results:
                if not physical_feat:
                    final_rank.append((phone,results[phone]*(1+min(2,1+feature_mat[phone_to_index[phone]][feat_to_index["price"]]/1000))))
                else:
                    final_rank.append((phone,results[phone]))
            final_rank = sorted(final_rank,key=lambda x: x[1])[::-1]
            result = []
            urls   = []
            for elt in final_rank:
                result.append(elt[0])
                urls.append(features[elt[0]][0])

            scores = []
            ml_prices = []
            for i in range(len(result)):
                scores.append(final_rank[i][1])
                ml_prices.append(feature_mat[phone_to_index[result[i]]][feat_to_index["price"]])

            feat_labels = {'3.5mm jack': 'Audio Jack',
                       'battery': 'Battery Size',
                       'card slot': 'SD Card Slot',
                       'cpu': 'Processor (CPU)',
                       'dual': 'Dual Camera',
                       'face': 'Face Unlock',
                       'finger': 'Fingerprint Sensor',
                       'front camera': 'Front Camera',
                       'ppi': 'Screen Resolution',
                       'ram': 'Memory (RAM)',
                       'rear camera': 'Rear Camera',
                       'sim': 'Dual SIM',
                       'size': 'Screen Size',
                       'storage': 'Internal Storage',
                       'thickness': 'Thinness',
                       'video': 'Video Quality',
                       'waterproof': 'Water Resistance',
                      }

            def plot_bar(phone, features, i):
                label = []
                feat_scores = []
                for f in features:
                    label.append(feat_labels[f])
                    feat_scores.append(feature_mat[phone_to_index[phone]][feat_to_index[f]])
                #plotting
                index = np.arange(len(label))
                plt.bar(index, feat_scores, color=('#ABCCD4'), edgecolor=('#ADD4D4'), linewidth=2)
                plt.xlabel('Selected Features', fontsize=5)
                plt.ylabel('Normalized Scores', fontsize=5)
                plt.xticks(index, label, fontsize=5)
                plt.ylim((0.0,1.0))
                plt.title('Scores of User Selected Features')
                plt.savefig('images/bar_%i.jpg' % (i+1), dpi=150, facecolor='#989898', edgecolor='#989898')
                plt.close()

            for i,phone in enumerate(result[:18]):
                plot_bar(phone, query_feat, i)

            def clean_phone(url, i):
                with Image(filename=url) as img:
                    img.format = 'png'
                    with Color('#FDFDFD') as white:
                        twenty_percent = int(65535 * 0.02)
                        img.transparent_color(white, alpha=0.0, fuzz=twenty_percent)
                    img.save(filename="images/clean_phone_%i.png" % (i+1)) #CHANGE TO DYNAMIC NAME OF PHONE RANKING

            for i,phone in enumerate(result[:18]):
                clean_phone(features[phone][0], i)

            print(result)
            return [result,urls,new_string,scores,ml_prices]

        final = main(budget,feature_list,condition)


        return render_template('search.html', name=project_name,netid=net_id, check=check, check2=check2, mate2=mate2, mate=mate, flag=flag, flag2=flag2,
                                condition=condition, names=final[0], urls = final[1],budget=str(budget), features=feature_list,close_words = final[2],scores=final[3],price=final[4])
