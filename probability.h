/*------------------------------------------------------------------------------
auth | Roberto Peribáñez Iglesias
date | dec. 2018
------------------------------------------------------------------------------*/

#ifndef PROBABILITY_H
#define PROBABILITY_H

#include "tree.h"


namespace ML
{
//------------------------------------------------------------------------| ProbabilityTree

class ProbabilityTree : public Tree
{
public :

    ProbabilityTree(void);

    Node *TreeInduction(DataFrame &subsamples, std::vector<std::wstring> subattributes);

    void Build(void);
};
}

#endif // PROBABILITY_H
