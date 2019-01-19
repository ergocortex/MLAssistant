/*------------------------------------------------------------------------------
auth | Roberto Peribáñez Iglesias (ergocortex) on Dec. 2018
------------------------------------------------------------------------------*/

#include <limits.h>

#include "association.h"

using namespace ML;

//------------------------------------------------------------------------| ItemSet

ItemSet::Item::Item(const MathOp mathop, const Variant &value, const std::vector <uint> &indexes) :
    mathop(mathop), value(value), indexes(indexes) {}

ItemSet::ItemSet(const float &p) : p(p) {}

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

AssociationRules::Completeness::Completeness(uint index, float p) : index(index), p(p) {}

float AssociationRules::Completeness::Calculate(void)
{
    float result = 0.0f;
    float delta = 1.0f / (float)(antecedents.size());

    for(uint i = 0, n = antecedents.size(); i < n; ++i)
    {
        if(antecedents[i]) result += delta;
    }

    return(result);
}

AssociationRules::AssociationRules(void) : support_threshold(3), confidence_threshold(0.9f) {}

void AssociationRules::Generator(ItemSet *source)
/*------------------------------------------------------------------------------
nots | . high attributes entropy with low support threshold leads to out of memory
------------------------------------------------------------------------------*/
{
    const int maxdeep = 0;

    static int deep = 0;

    ++deep;

    for(uint i = 0, n = samples.attributes.size(); i < n; ++i)
    {
        if(source->itemmap.find(samples.attributes[i]->name) != source->itemmap.end()) continue;

        std::vector<ML::Attribute::ProbabilityDistribution> *probabilityDistribution = nullptr;

        // '--> if itemmap is empty calculate distribution of all elements.

        if(source->itemmap.empty())
            probabilityDistribution = samples.attributes[i]->GetProbabilityDistribution();
        else
            probabilityDistribution = samples.attributes[i]->GetProbabilityDistribution(source->GetRestrictiveItem());

        for(auto it = probabilityDistribution->end(); it != probabilityDistribution->begin();  --it)
        {
            auto &distribution = *(it - 1);

            if(distribution.value.IsNull() || (distribution.indexes.size() < support_threshold))
                probabilityDistribution->erase(it - 1);
        }

        if(probabilityDistribution->empty()) continue;

        for(auto &distribution : *probabilityDistribution)
        {
            itemSet.push_back(new ItemSet(*source));

            itemSet.back()->p = distribution.p;

            itemSet.back()->itemmap.insert(std::pair<std::wstring, ItemSet::Item>(samples.attributes[i]->name,
                ItemSet::Item(distribution.mathop, distribution.value, distribution.indexes)));

            if((maxdeep == 0) || (deep < maxdeep))
            {
                Generator(itemSet.back());
            }
        }

        delete(probabilityDistribution);
    }

    --deep;
}

float AssociationRules::CalcConfidence(ItemSet *itemSet, uint mask)
{
    std::vector <uint> restrictions;

    // '--> antecedents (ones)

    restrictions.clear();

    for(uint i = 0, n = itemSet->itemmap.size(); i < n; ++i)
    {
        if(mask & (1 << i))
            restrictions.push_back(i);
    }

    float antecedends = itemSet->GetOverlapping(restrictions);

    // '--> consequents (zeros)

    restrictions.clear();

    for(uint i = 0, n = itemSet->itemmap.size(); i < n; ++i)
    {
        if(!(mask & (1 << i)))
            restrictions.push_back(i);
    }

    float consequents = itemSet->GetOverlapping(restrictions);

    return(min(consequents / antecedends, 1.0f));
}

void AssociationRules::CreateRule(ItemSet *itemSet, uint mask, float p)
{
    std::map<std::wstring, ItemSet::Item>::iterator it;

    rules.push_back(new Rule());

    // '--> antecedents (ones)

    it = itemSet->itemmap.begin();

    for(uint i = 0, n = itemSet->itemmap.size(); i < n; ++i, ++it)
    {
        if(mask & (1 << i))
            rules.back()->antecedents.push_back(Rule::Factor(it->first, it->second.mathop, it->second.value));
    }

    // '--> consequents (zeros)

    it = itemSet->itemmap.begin();

    for(uint i = 0, n = itemSet->itemmap.size(); i < n; ++i, ++it)
    {
        if(!(mask & (1 << i)))
            rules.back()->consequents.push_back(Rule::Factor(it->first, it->second.mathop, it->second.value));
    }

    rules.back()->p = p;
}

void AssociationRules::Build(void)
/*------------------------------------------------------------------------------
nots | . confidence is the probability of the rule.
------------------------------------------------------------------------------*/
{
    itemSet.clear();
    rules.clear();

    // '--> ItemSet Generation

    itemSet.push_back(new ItemSet());

    Generator(itemSet.back());

    // '--> Rule Generation

    for(uint i = 0, n = itemSet.size(); i < n; ++i)
    {
        // '--> combinations of bits with neither all zeroes nor all ones.

        int bits = itemSet[i]->itemmap.size();

        uint combinations = max(0, pow(2, bits) - 2);

        for(uint mask = 1; mask <= combinations; ++mask)
        {
            float confidence = CalcConfidence(itemSet[i], mask);

            if(confidence > confidence_threshold)
                CreateRule(itemSet[i], mask, confidence);
        }
    }
}

AssociationRules::Completeness *AssociationRules::Predict(DataFrame &sample)
{
    if(rules.empty()) return(nullptr);

    std::vector <Completeness> completeness;

    // '--> Populate

    for(uint i = 0, n = rules.size(); i < n; ++i)
    {
        completeness.push_back(Completeness(i, rules[i]->p));

        for(uint j = 0, m = rules[i]->antecedents.size(); j < m; ++j)
        {
            std::wstring attribute = rules[i]->antecedents[j].attribute;
            uint index = sample.GetColumnByAttribute(attribute);

            Variant valueA = sample.attributes[index]->GetCell(0);
            Variant valueB = rules[i]->antecedents[j].value;

            if(Validate(valueA, rules[i]->antecedents[j].mathop, valueB))
                completeness.back().antecedents.push_back(true);
            else
                completeness.back().antecedents.push_back(false);
        }
    }

    // '--> Priorize

    for(uint i = 1, n = completeness.size(); i < n; ++i)
    {
        float reference = completeness[0].Calculate();
        float complete = completeness[i].Calculate();

        if(complete > reference)
            std::swap(completeness[i], completeness[0]);
        else
        {
            if(fcmp(complete, reference) && (completeness[i].p > completeness[0].p))
                std::swap(completeness[i], completeness[0]);
        }
    }

    Completeness *result = new Completeness(completeness[0]);

    return(result);
}
