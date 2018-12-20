#include "rule.h"

using namespace ML;

//------------------------------------------------------------------------| Rule

Rule::Factor::Factor(const std::wstring &attribute, const Variant &restriction) :
    attribute(attribute), restriction(restriction) {}

Rule::Rule(void) {}
