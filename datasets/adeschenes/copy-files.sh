
# Beta2-Spectrin
FILES=(
    "adeschenes/2023-04-25_CalgarySampleTests/UnmixingControl_msB2Spectrin_AF594_1.msr"
    "adeschenes/2023-04-25_CalgarySampleTests/UnmixingControl_msB2Spectrin_AF594_2.msr"
    "adeschenes/2023-04-25_CalgarySampleTests/UnmixingControl_msB2Spectrin_AF594_3.msr"
)
for file in "${FILES[@]}"
do
    rclone copyto --progress --update "valeria-s3:flclab-abberior-sted/${file}" "valeria-s3:flclab-private/FLCDataset/${file}"
done

# Vimentin
FILES=(
    "adeschenes/2023-12-04/MsVimentin_1to100_GAM_STOrange_1to250_Culture80K_FixPFA20min_DIV8_cs1_c1.msr"
    "adeschenes/2023-12-04/MsVimentin_1to100_GAM_STOrange_1to250_Culture80K_FixPFA20min_DIV8_cs1_c2.msr"
    "adeschenes/2023-12-04/MsVimentin_1to100_GAM_STOrange_1to250_Culture80K_FixPFA20min_DIV8_cs1_c3.msr"
    "adeschenes/2023-11-28_TestAnticorpsVimentin_CultureGlie/MsVimentin_1to100_GAM_STOrange_1to250_Culture80K_FixPFA20min_DIV8_cs1_c1.msr"
    "adeschenes/2023-11-28_TestAnticorpsVimentin_CultureGlie/MsVimentin_1to100_GAM_STOrange_1to250_Culture80K_FixPFA20min_DIV8_cs1_c2.msr"
    "adeschenes/2024-02-13_msVimentinAntibody_DilutionTest/MsVimentin_1to100_GAM_STOrange_1to250_CultureCortex80Kcells_FixPFA20min_DIV8_cs1_c1.msr"
    "adeschenes/2024-02-13_msVimentinAntibody_DilutionTest/ MsVimentin_1to100_GAM_STOrange_1to250_CultureCortex80Kcells_FixPFA20min_DIV8_cs1_c1_2.msr"
    "adeschenes/2024-02-13_msVimentinAntibody_DilutionTest/MsVimentin_1to100_GAM_STOrange_1to250_CultureCortex80Kcells_FixPFA20min_DIV8_cs1_c2.msr"
    "adeschenes/2024-02-13_msVimentinAntibody_DilutionTest/MsVimentin_1to100_GAM_STOrange_1to250_CultureCortex80Kcells_FixPFA20min_DIV8_cs1_c3.msr"
    "adeschenes/2024-02-13_msVimentinAntibody_DilutionTest/MsVimentin_1to250_GAM_STOrange_1to250_CultureCortex80Kcells_FixPFA20min_DIV8_cs1_c1.msr"
    "adeschenes/2024-02-13_msVimentinAntibody_DilutionTest/MsVimentin_1to250_GAM_STOrange_1to250_CultureCortex80Kcells_FixPFA20min_DIV8_cs1_c2.msr"
    "adeschenes/2024-02-13_msVimentinAntibody_DilutionTest/MsVimentin_1to250_GAM_STOrange_1to250_CultureCortex80Kcells_FixPFA20min_DIV8_cs1_c3.msr"
    "adeschenes/2024-02-13_msVimentinAntibody_DilutionTest/MsVimentin_1to250_GAM_STOrange_1to250_CultureCortex80Kcells_FixPFA20min_DIV8_cs1_c4.msr"
    "adeschenes/2024-02-13_msVimentinAntibody_DilutionTest/MsVimentin_1to250_GAM_STOrange_1to250_CultureCortex80Kcells_FixPFA20min_DIV8_cs1_c5.msr"
    "adeschenes/2024-02-13_msVimentinAntibody_DilutionTest/MsVimentin_1to250_GAM_STOrange_1to250_CultureCortex80Kcells_FixPFA20min_DIV8_cs1_c5_2.msr"
    "adeschenes/2024-02-13_msVimentinAntibody_DilutionTest/MsVimentin_1to250_GAM_STOrange_1to250_CultureCortex80Kcells_FixPFA20min_DIV8_cs1_c6_1.msr"
)
for file in "${FILES[@]}"
do
    rclone copyto --progress --update "valeria-s3:flclab-abberior-sted/${file}" "valeria-s3:flclab-private/FLCDataset/${file}"
done