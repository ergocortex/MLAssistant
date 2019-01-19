#include "rule.h"

using namespace ML;

//------------------------------------------------------------------------| Rule

Rule::Factor::Factor(const std::wstring &attribute, MathOp mathop, const Variant &restriction) :
    attribute(attribute), mathop(mathop), value(restriction) {}

Rule::Rule(const float p) : p(p) {}
