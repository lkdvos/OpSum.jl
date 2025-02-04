"verify if this token represents the starting state"
isbegin(O::Enum) = O == first(instances(typeof(O)))
"verify if this token represents the ending state"
isend(O::Enum) = O == last(instances(typeof(O)))


