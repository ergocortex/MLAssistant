/*------------------------------------------------------------------------------
auth | Roberto Peribáñez Iglesias (ergocortex) on Dec. 2018
------------------------------------------------------------------------------*/

#include <limits.h>

#include "association.h"

using namespace ML;

//------------------------------------------------------------------------| ItemSet

ItemSet::Item::Item(const Variant &value, const std::vector <uint> &indexes) :
    value(value), indexes(indexes) {}

ItemSet::ItemSet(void) {}

std::vector <uint> &ItemSet::GetRestrictiveItem(const std::vector <uint> &restrictions)
{
    std::map<std::wstring, Item>::iterator restrictive = itemmap.end();

    uint size = std::numeric_limits<int>::max();

    uint i = 0;

    for(auto it = itemmap.begin(); it != itemmap.end(); ++it, ++i)
    {
        if(!restrictions.empty() && (std::find(restrictions.begin(), restrictions.end(), i) == restrictions.end()))
            continue;

        if(it->second.indexes.size() < size)
        {
            size = it->second.indexes.size();
            restrictive = it;
        }
    }

    return(restrictive->second.indexes);
}

uint ItemSet::GetOverlapping(const std::vector <uint> &restrictions)
{
    return(GetRestrictiveItem(restrictions).size());
}

//------------------------------------------------------------------------| AssociationRules

AssociationRules::AssociationRules(void) : support_threshold(3), confidence_threshold(0.9f) {}

void AssociationRules::Generator(ItemSet &source)
{
    for(uint i = 0, n = samples.attributes.size(); i < n; ++i)
    {
        if(source.itemmap.find(samples.attributes[i]->name) == source.itemmap.end()) continue;

        std::vector<ML::Attribute::ProbabilityDistribution> *probabilityDistribution = nullptr;

        probabilityDistribution = samples.attributes[i]->GetProbabilityDistribution(source.GetRestrictiveItem());

        for(auto it = probabilityDistribution->end(); it != probabilityDistribution->begin();  --it)
        {
            auto &value = *(it - 1);

            if(value.indexes.size() < support_threshold)
                probabilityDistribution->erase(it - 1);
        }

        for(auto it = probabilityDistribution->begin(); it != probabilityDistribution->end(); ++i)
        {
            itemSet.push_back(ItemSet(source));

            itemSet.back().itemmap.insert(std::pair<std::wstring, ItemSet::Item>(samples.attributes[i]->name,
                ItemSet::Item((*probabilityDistribution)[i].value, (*probabilityDistribution)[i].indexes)));

            Generator(itemSet.back());
        }
    }
}

float AssociationRules::CalcConfidence(ItemSet &itemSet, uint mask)
{
    std::vector <uint> restrictions;

    // '--> antecedents (ones)

    restrictions.clear();

    for(uint i = 0, n = itemSet.itemmap.size(); i < n; ++i)
    {
        if(mask & (1 << i))
            restrictions.push_back(i);
    }

    float antecedends = itemSet.GetOverlapping(restrictions);

    // '--> consequents (zeros)

    restrictions.clear();

    for(uint i = 0, n = itemSet.itemmap.size(); i < n; ++i)
    {
        if(!(mask & (1 << i)))
            restrictions.push_back(i);
    }

    float consequents = itemSet.GetOverlapping(restrictions);

    return(consequents / antecedends);
}

void AssociationRules::CreateRule(ItemSet &itemSet, uint mask)
{
    std::map<std::wstring, ItemSet::Item>::iterator it;

    rules.push_back(Rule());

    // '--> antecedents (ones)

    it = itemSet.itemmap.begin();

    for(uint i = 0, n = itemSet.itemmap.size(); i < n; ++i, ++it)
    {
        if(mask & (1 << i))
            rules.back().antecedents.push_back(Rule::Factor(it->first, it->second.value));
    }

    // '--> consequents (zeros)

    it = itemSet.itemmap.begin();

    for(uint i = 0, n = itemSet.itemmap.size(); i < n; ++i, ++it)
    {
        if(!(mask & (1 << i)))
            rules.back().consequents.push_back(Rule::Factor(it->first, it->second.value));
    }
}

void AssociationRules::Build(void)
{
    itemSet.clear();
    rules.clear();

    // '--> ItemSet Generation

    for(uint i = 0, n = samples.attributes.size(); i < n; ++i)
    {
        std::vector<ML::Attribute::ProbabilityDistribution> *probabilityDistribution = nullptr;

        probabilityDistribution = samples.attributes[i]->GetProbabilityDistribution();

        for(auto it = probabilityDistribution->end(); it != probabilityDistribution->begin();  --it)
        {
            auto &value = *(it - 1);

            if(value.indexes.size() < support_threshold)
                probabilityDistribution->erase(it - 1);
        }

        for(auto it = probabilityDistribution->begin(); it != probabilityDistribution->end(); ++i)
        {
            itemSet.push_back(ItemSet());

            itemSet.back().itemmap.insert(std::pair<std::wstring, ItemSet::Item>(samples.attributes[i]->name,
                ItemSet::Item((*probabilityDistribution)[i].value, (*probabilityDistribution)[i].indexes)));

            Generator(itemSet.back());
        }
    }

    // '--> Rule Generation

    for(uint i = 0, n = itemSet.size(); i < n; ++i)
    {
        for(uint mask = 1, combinations = (2 ^ itemSet[i].itemmap.size()) - 1; mask < combinations; ++mask)
        {
            if(CalcConfidence(itemSet[i], mask) > confidence_threshold)
                CreateRule(itemSet[i], mask);
        }
    }
}
