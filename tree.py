
import nltk
a = nltk.Tree.fromstring("(TOP (S (NP (_ I)) (VP (_ drank) (NP (NP (_ a) (_ cup)) (PP (_ of) (NP (_ water))) (SBAR (WHNP (_ that)) (S (VP (_ has) (NP (_ some) (_ juice)) (PP (_ in) (NP (_ it)))))))) (_ .)))")
def FindPhrase(tree,phrase_label):
    count = 0
    flag = False
    if tree.label() == '_': # end of tree
            #print('end of tree')
            return 0
    
    children = [tree[j].label() for j in range(len(tree))]

    if tree.label() == phrase_label or (phrase_label == 'ADVP' and tree.label() in phrase_label): 
        if len(tree) >1 :
            count += 1 # found target phrase
        if not phrase_label in children : # do not search XP in XP, except XP does not have XP as its direct children, or XP has a SBAR child
            pass 
        elif 'SBAR' in children and phrase_label in children:
            flag = True # continue searching, but do not search XP's direct child XP
        else:
            return count # stop searching
    for i in range(len(tree)):
        if flag and tree[i].label() == phrase_label:
            continue
        count += FindPhrase(tree[i],phrase_label)      
    return count
        
        
c = FindPhrase(a,'NP')
print(c)