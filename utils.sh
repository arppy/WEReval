

patterns=("C_001" "C_003" "C_006" "C_010" "C_016" "C_018" "C_026" "C_027" "C_029" "C_031" "C_033" "C_036" "C_040" "C_041" "C_042" "C_043" "C_048")
zip -r HunDysSepFormerDNS4.zip $(for pattern in "${patterns[@]}"; do echo HunDysSepFormerDNS4_wav/${pattern}*; done)