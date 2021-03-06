/*------------------------------------------------------------------------------
auth | Roberto Peribáñez Iglesias (ergocortex) 2018
------------------------------------------------------------------------------*/

#include "tree.h"
#include "probability.h"

using namespace ML;

//------------------------------------------------------------------------| ProbabilityTree

ProbabilityTree::ProbabilityTree(ubyte attributeSelection) : Tree(), attributeSelection(attributeSelection) {}

ML::Node *ProbabilityTree::TreeInduction(DataFrame &subsamples, std::vector <std::wstring> subattributes)
{
    Node *node = AddNode();

    bool leaf = (subattributes.size() == 1);

    std::wstring attribute;

    switch(attributeSelection)
    {
    case 0 : attribute = AttributeSelection::InformationGain(subsamples, subattributes, leaf ? 0 : 1); break;
    case 1 : attribute = AttributeSelection::GiniImpurity(subsamples, subattributes, leaf ? 0 : 1); break;
    case 2 : attribute = AttributeSelection::ProportionGain(subsamples, subattributes, leaf ? 0 : 1); break;
    }

    ClearAttribute(attribute, subattributes);

    node->data = attribute;

    for(Attribute *factor : subsamples.attributes)
    {
        if(factor->name == attribute)
        {
            std::vector<ML::Attribute::ProbabilityDistribution> *probabilityDistribution = factor->GetProbabilityDistribution();

            for(uint i = 0, n = (*probabilityDistribution).size(); i < n; ++i)
            {
                Node *child = nullptr;

                if(leaf)
                {
                    child = AddNode();

                    child->leaf = true;
                    child->data = (*probabilityDistribution)[i].value;
                }
                else
                {
                    child = TreeInduction(*subsamples.GetSubDataFrame((*probabilityDistribution)[i].indexes), subattributes);
                }

                if(child) AddEdge((*probabilityDistribution)[i].value, (*probabilityDistribution)[i].p,
                                  (*probabilityDistribution)[i].mathop, node, child);
            }

            delete(probabilityDistribution);
        }
    }

    return(node);
}

void ProbabilityTree::Build(void)
{
    DataFrame subsamples = samples;

    std::vector <std::wstring> subattributes;

    for(uint i = 0, n = samples.attributes.size(); i < n; ++i)
        subattributes.push_back(samples.attributes[i]->name);

    clrptrvector<Node *>(nodes);
    clrptrvector<Edge *>(edges);

    TreeInduction(subsamples, subattributes);

    RankHierarchy();
}
