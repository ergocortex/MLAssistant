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

        MathOp mathop;
        Variant value;

        std::vector <uint> indexes;

    public :

        Item(const MathOp mathop, const Variant &value, const std::vector <uint> &indexes);
    };

public :

    std::map <std::wstring, Item> itemmap;

    float p;

public :

    ItemSet(const float &p = 0.0f);

    std::vector<uint> &GetRestrictiveItem(const std::vector<uint> &restrictions = {});
    uint GetOverlapping(const std::vector <uint> &restrinction);
};

//------------------------------------------------------------------------| AssociationRules

class AssociationRules
{
public :

    struct Completeness
    {
    public :

        uint index;

        float p;

        std::vector <bool> antecedents;

    public :

        Completeness(uint index, float p);

        float Calculate(void);
    };

public :

    DataFrame samples;

    std::vector <ItemSet *> itemSet;
    std::vector <Rule *> rules;

    int   support_threshold;
    float confidence_threshold;

public :

    AssociationRules(void);

    void Generator(ItemSet *source);
    float CalcConfidence(ItemSet *itemSet, uint mask);
    void CreateRule(ItemSet *itemSet, uint mask, float p);

    void Build(void);

    Completeness *Predict(DataFrame &sample);
};
}

#endif // ASSOCIATION_H
