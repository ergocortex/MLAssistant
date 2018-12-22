#include "rule.h"

using namespace ML;

//------------------------------------------------------------------------| Rule

Rule::Factor::Factor(const std::wstring &attribute, ubyte mathop, const Variant &restriction) :
    attribute(attribute), mathop(mathop), restriction(restriction) {}

Rule::Rule(void) {}
