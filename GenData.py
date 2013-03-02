import Whale as w
import matplotlib.pylab as pl

no_whale_mean = w.calculate_mean_over_class(cases="no_whales");
whale_mean = w.calculate_mean_over_class(cases="whales")

# pl.figure(1)
# pl.subplot(121)
# pl.imshow(no_whale_mean, aspect='auto')
# pl.title("mean spectrogram: no whale")
# pl.subplot(122)
# pl.imshow(whale_mean, aspect='auto')
# pl.title("mean spectrogram: whale")
# pl.show()

projection_vec = whale_mean - no_whale_mean
translation_vec = no_whale_mean
no_whale_projected = w.translate_and_project_onto_vector(
    cases=w.no_whale_cases,
    t_vec=translation_vec,
    p_vec=projection_vec
    )
whale_projected = w.translate_and_project_onto_vector(
    cases=w.whale_cases,
    t_vec=translation_vec,
    p_vec=projection_vec
    )

test_projected = w.translate_and_project_onto_vector(
    cases=w.test_cases,
    t_vec=translation_vec,
    p_vec=projection_vec,
    training=False
    )

# pl.figure(2)
# pl.imshow(projection_vec, aspect='auto')
# pl.show()

projection_file = open('projections.txt', 'w')
projection_file.write("Projected_Spectrogram Is_Whale?\n")
for item in no_whale_projected:
  projection_file.write("%s 0\n" % item)
for item in whale_projected:
  projection_file.write("%s 1\n" % item)
projection_file.close()

testCsv_file = open('testRes.csv', 'w')
for item in test_projected:
  testCsv_file.write("%s\n" % item)
testCsv_file.close()
