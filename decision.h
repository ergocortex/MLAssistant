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
{
public :

    DecisionTree(void);

    Node *TreeInduction(DataFrame &subsamples, std::vector <std::wstring> &subattributes);

    void Build(void);

    void KCrossValidation(uint k);
};
}

#endif // DECISION_H
