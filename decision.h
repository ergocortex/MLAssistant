/*------------------------------------------------------------------------------
auth | Roberto Peribáñez Iglesias
date | dec. 2018
------------------------------------------------------------------------------*/

#ifndef DECISION_H
#define DECISION_H

#include "tree.h"


namespace ML
{
//------------------------------------------------------------------------| DecisionTree

class DecisionTree : public Tree
/*------------------------------------------------------------------------------
vars | attributeSelection | 0 : Information Gain | 1 : Gini Impurity | 2 : Proportion Gain
------------------------------------------------------------------------------*/
{
public :

    DataFrame confusionMatrix;

    ubyte attributeSelection;

public :

    static int GetArgumentIndex(const std::wstring &value, uint index);

public :

    DecisionTree(ubyte attributeSelection = 0);

    void Train(const DataFrame *dataframe = nullptr);
    void KCrossValidation(uint k);  

private :

    Node *TreeInduction(DataFrame &subsamples, std::vector <std::wstring> &subattributes);

    int GetConfusionIndex(const std::wstring &value);    
};
}

#endif // DECISION_H
