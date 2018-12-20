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
        Variant restriction;

    public :

        Factor(const std::wstring &attribute, const Variant &restriction);
    };

public :

    std::vector <Factor> antecedents;
    std::vector <Factor> consequents;

public :

    Rule(void);
};
}

#endif // RULE_H
