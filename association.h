/*------------------------------------------------------------------------------
auth | Roberto Peribáñez Iglesias (ergocortex) 2018
------------------------------------------------------------------------------*/

#ifndef ASSOCIATION_H
#define ASSOCIATION_H

#include "core.h"
#include "rule.h"

namespace ML
{
//------------------------------------------------------------------------| ItemSet

class ItemSet
{
public :

    struct Item
    {
    public :

        Variant value;

        std::vector <uint> indexes;

    public :

        Item(const Variant &value, const std::vector <uint> &indexes);
    };

public :

    std::map <std::wstring, Item> itemmap;

public :

    ItemSet(void);

    std::vector <uint> &GetRestrictiveItem(const std::vector<uint> &restrictions = {});
    uint GetOverlapping(const std::vector <uint> &restrinction);
};

//------------------------------------------------------------------------| AssociationRules

class AssociationRules
{
public :

    DataFrame samples;

    std::vector <ItemSet> itemSet;
    std::vector <Rule> rules;

    int   support_threshold;
    float confidence_threshold;

public :

    AssociationRules(void);

    void Generator(ItemSet &source);
    float CalcConfidence(ItemSet &itemSet, uint mask);
    void CreateRule(ItemSet &itemSet, uint mask);

    void Build(void);
};
}

#endif // ASSOCIATION_H
