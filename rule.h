#ifndef RULE_H
#define RULE_H

#include <string>
#include <vector>

#include "core.h"

namespace ML
{
//------------------------------------------------------------------------| Rule

class Rule
{
public :

    struct Factor
    {
    public :

        std::wstring attribute;
        MathOp mathop;
        Variant value;

    public :

        Factor(const std::wstring &attribute, MathOp mathop, const Variant &value);
    };

public :

    std::vector <Factor> antecedents;
    std::vector <Factor> consequents;

    float p;

public :

    Rule(const float p = 0.0f);
};
}

#endif // RULE_H
