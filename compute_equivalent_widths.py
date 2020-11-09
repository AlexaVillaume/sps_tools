import sys
import subprocess
import glob
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from astropy.table import Table, Column, hstack, vstack
from astropy.io import fits
from scipy import stats
#import astropy.stats as stats
from astropy.convolution import convolve
from astropy.convolution import Gaussian1DKernel
from sps_basis import StarBasis
#import database_utils as dbu
#import make_plots
import read_mist_models as read_mist

class Feature(object):

    def __init__(self, line, line_num):
        cols = line.split(',')
        if len(cols) != 9:
            error = ("Not enough information to "
                     "compute equivalent width on line {}")
            raise ValueError(error.format(line_num+1))
        else:
            self.name = cols[0]
            self.central = float(cols[1])
            self.feature = [float(cols[2]), float(cols[3])]
            self.blue_c  = [float(cols[4]), float(cols[5])]
            self.red_c   = [float(cols[6]), float(cols[7])]
            self.flag = float(cols[8])
            self.equiv_w = None
            self.equiv_w_upper = None
            self.equiv_w_lower = None

    def get_ew(self, wave, sample):
        def int4ind(wave, spec, low, high):
            # Need to account for the fact that I might not have
            # exact correspondance of the feature defintions to my
            # wavelength grid.

            # Indices of wavelength values closest to my feature definition
            l1 = (np.abs(wave - low)).argmin() - 1
            l2 = (np.abs(wave - high)).argmin() - 1

            # Get flux at exact feature definitions through linear interpolation
            f1 = (spec[l1+1]-spec[l1])/(wave[l1+1]-wave[l1])*(low-wave[l1])+spec[l1]
            f2 = (spec[l2+1]-spec[l2])/(wave[l2+1]-wave[l2])*(high-wave[l2])+spec[l2]

            # Integrate
            int4ind = sum((wave[l1+2:l2+1]-wave[l1+1:l2])*(spec[l1+2:l2+1]+spec[l1+1:l2])/2.)
            int4ind = int4ind + (wave[l1+1]-low)*(f1 + spec[l1+1])/2.
            int4ind = int4ind + (high-wave[l2])*(f2 + spec[l2])/2.

            return int4ind

        if len(sample) == 1:
            spec = sample[0]

            # Define blue continuum
            cb = int4ind(wave, spec, self.blue_c[0], self.blue_c[1])
            cb = cb/(self.blue_c[1]-self.blue_c[0])
            lb = round((self.blue_c[0]+self.blue_c[1])/2., 4)

            # Define red contiuum
            cr = int4ind(wave, spec, self.red_c[0], self.red_c[1])
            cr = cr/(self.red_c[1]-self.red_c[0])
            lr = round((self.red_c[0]+self.red_c[1])/2., 4)

            # Compute the integrate (flux_feature/flux_continuum)
            # flux_continuum is a linear interpolation between cb and cr
            int_fifc = int4ind(wave, spec/((cr-cb)/(lr-lb)*(wave-lb)+cb),
                               self.feature[0], self.feature[1])

            self.equiv_w = (self.feature[1] - self.feature[0]) - int_fifc

        elif len(sample) > 1:
            distro = []
            for spec in sample:
                # Define blue continuum
                cb = int4ind(wave, spec, self.blue_c[0], self.blue_c[1])
                cb = cb/(self.blue_c[1]-self.blue_c[0])
                lb = round((self.blue_c[0]+self.blue_c[1])/2., 4)

                # Define red contiuum
                cr = int4ind(wave, spec, self.red_c[0], self.red_c[1])
                cr = cr/(self.red_c[1]-self.red_c[0])
                lr = round((self.red_c[0]+self.red_c[1])/2., 4)

                # Compute the integrate (flux_feature/flux_continuum)
                # flux_continuum is a linear interpolation between cb and cr
                int_fifc = int4ind(wave, spec/((cr-cb)/(lr-lb)*(wave-lb)+cb),
                                   self.feature[0], self.feature[1])
                if self.flag == 2:
                    distro.append((self.feature[1] - self.feature[0]) - int_fifc)
                elif self.flag == 1:
                    distro.append(-2.5*np.log10(int_fifc/(self.feature[1] - self.feature[0])))

            distro = np.array(distro)
            mean, sigma = distro.mean(), distro.std()
            conf_int_a = stats.norm.interval(0.68, loc=mean, scale=sigma)
            self.equiv_w = mean
            self.equiv_w_lower, self.equiv_w_upper = conf_int_a
        else:
            error = ('Valid spectrum not given')
            raise ValueError(error)

def get_equiv_widths(linelist, star, **keywords):

    if 'error' in keywords.keys() and keywords['error']:
        star.spec = np.ma.masked_where(star.spec <= 0, star.spec)
        star.unc = np.ma.masked_where(star.unc <= 0, star.unc)

        sample = np.zeros((100, len(star.wave)))
        for i, (flux, unc) in enumerate(zip(star.spec, star.unc)):
            sample[:,i] = (np.random.normal(flux, scale=unc, size=100))
    else:
        sample = np.array([star.spec])

    names, strengths = [], []
    upper, lower = [], []
    with open(linelist) as f:
        for i, line in enumerate(f):
            line = Feature(line, i)
            line.get_ew(star.wave, sample)
            strengths.append(line.equiv_w)
            names.append(line.name)
            if line.equiv_w_upper:
                upper.append(line.equiv_w_upper)
            else:
                upper.append(0.)
            if line.equiv_w_lower:
                lower.append(line.equiv_w_lower)
            else:
                lower.append(0.)

    return names, strengths, upper, lower

#class Library(object):
#    def __init__(self, wave):
#        self.wave = wave

class Star(object):
    def __init__(self, name, teff, logg, spec, wave):
        #Library.__init__(self.wave)
        self.name = name
        self.teff = teff
        self.logg = logg
        self.spec = spec
        self.wave = wave
        self.unc = None

        #super(Star, self).__init__()

    def add_uncertainty(self, unc):
        self.unc = unc

    def print_attributes(self):
        print self.name, self.teff, self.logg, self.spec


def write_equiv_widths(linelist, stars, outfile):

    with open(outfile, 'w') as f:
        f.write("# teff logg ")
        with open(linelist) as ll:
            for row in ll:
                f.write("{:15}".format(row.split(',')[0]))
        f.write("\n")

        for i, star in enumerate(stars):
            line_names, line_strengths, upper, lower = get_equiv_widths(linelist, star)

            f.write("{0:1.4} {1:5.5}".format(star.teff, star.logg))
            for value in line_strengths:
                if isinstance(value, np.float64):
                    f.write("{0:15.5f}".format(value))
                else:
                    print "something wrong: {}".format(value)
                    f.write("{0:15.5f}".format(-1))

            f.write("\n")

def old_irtf_lib(linelist):

    old_irtf = Table.read('OldLibrary/old_library_irtf_final.fits', format='fits')
    with fits.open('OldLibrary/old_library_irtf_final.fits') as tmp:
        wavelength = (tmp[2].data) # Angstrom

    outfile = 'test_old_irtf_lib_ews.txt'

    stars = []
    for star in old_irtf:
        stars.append(Star(star['NAME'], star['TEFF'],
                          star['LOGL'], star['SPEC'], wavelength))

    write_equiv_widths(linelist, stars, outfile)

def new_irtf_lib(linelist, **keywords):
    sqlite_file = 'stellar_library.db'
    connection, cursor = dbu.connect(sqlite_file)
    #query = ('SELECT * FROM targets WHERE IRTF_spec IS NOT NULL '
    #         'AND logl IS NOT NULL AND ShapeIssue IS NULL '
    #         'AND ParamIssue IS NULL AND QualityIssue IS NULL')
    query = ('SELECT * FROM mdwarfs')
    cursor.execute(query)
    new_irtf = cursor.fetchall()

    if not keywords['model']:
        outfile = 'test_new_irtf_lib_ews.txt'
    else:
        outfile = 'test_new_irtf_lib_c3k_ews.txt'

    for i, star in enumerate(new_irtf):
        #result = make_plots.MILESlib(star)
        result = make_plots.MDwarf(star)
        if result.modified is None:
            continue

        result.prep_data()
        if not keywords['model']:
            ## Mann Mdwarf
            obj = Star(star['Name'], star['teff'],
                         star['logg'], result.modified['flux'],
                         result.modified['wave']*1e4)
            #obj.add_uncertainty(result.modified['unc'])

            ## IRTF
            #spec = fits.open(star['full_spec'])
            #obj = Star(star['Name'], star['teff_prug'],
            #             star['logg_prug'], spec[1].data['flux'],
            #             spec[1].data['wavelength']*1e4)
            #obj.add_uncertainty(spec[1].data['uncertainty'])
        else:
            obj = Star(star['Name'], star['teff_prug'],
                         star['logg_prug'], result.c3k['flux3'],
                         result.c3k['wave']*1e4)

        names, strengths, upper, lower = get_equiv_widths(linelist, obj, error=False)
        print strengths, upper, lower

        cursor.execute("INSERT OR IGNORE INTO equiv_widths (Name) Values (?)",
                        (star['Name'],))
        cursor.execute("INSERT OR IGNORE INTO ew_upper (Name) Values (?)",
                        (star['Name'],))
        cursor.execute("INSERT OR IGNORE INTO ew_lower (Name) Values (?)",
                        (star['Name'],))
        for n, s, u, l in zip(names, strengths, upper, lower):
            try:
                cursor.execute('ALTER TABLE equiv_widths ADD COLUMN {}'.format(n))
            except:
                pass # Ignore the error thrown if column already exits
            try:
                cursor.execute('ALTER TABLE ew_upper ADD COLUMN {}_u'.format(n))
            except:
                pass # Ignore the error thrown if column already exits
            try:
                cursor.execute('ALTER TABLE ew_lower ADD COLUMN {}_l'.format(n))
            except:
                pass # Ignore the error thrown if column already exits

            command = ("UPDATE equiv_widths SET {}=? WHERE Name=?".format(n))
            cursor.execute(command, (s, star['Name']))

            command = ("UPDATE ew_upper SET {}_u=? WHERE Name=?".format(n))
            cursor.execute(command, (u, star['Name']))

            command = ("UPDATE ew_lower SET {}_l=? WHERE Name=?".format(n))
            cursor.execute(command, (l, star['Name']))

            connection.commit()
        print 'Star: {}'.format(i)

    dbu.close(connection)


def galaxies(linelist):

    files = ['M32a_for_ew_data.txt',
             'M32a_cen_for_ew_data.txt']


    for fname in files:
        name = fname.replace('_for_ew_data.txt', '')
        print name
        data = np.loadtxt(fname)
        obj = Star(name, name, name, data[:,1], data[:,0])
        obj.add_uncertainty(data[:,2])
        names, strengths, upper, lower = get_equiv_widths(linelist, obj, error=True)

        t = Table([names, strengths, upper, lower],
                    names=('Lines', 'Strengths', 'Upper', 'Lower'))

        t.write('{0}_equiv_widths_data.txt'.format(name),
                format='ascii.commented_header')

def mist_isochrone(linelist):
    from spi_stuff import spi
    sps = StarBasis('../PSI_plots/ckc+dMall_miles+irtf.forpsi.h5',
                    n_neighbors=1, verbose=True)

    isochrone_path = ('../MistIsochrones/'
                      'IRTF_Metallicities/')
    mist_files = glob.glob(isochrone_path + 'MIST*')
    metallicities = [0.0, 0.3, -0.5, -1.0, -1.5]

    # Overlap of cool dwarf and warm dwarf training sets
    d_teff_overlap = np.linspace(3000, 5500, num=100)
    d_weights = np.linspace(1, 0, num=100)

    # Overlap of warm giant and hot star training sets
    gh_teff_overlap = np.linspace(5500, 6500, num=100)
    gh_weights = np.linspace(1, 0, num=100)

    # Overlap of warm giant and cool giant training sets
    gc_teff_overlap = np.linspace(3500, 4500, num=100)
    gc_weights = np.linspace(1, 0, num=100)

    isochrones = []
    for iso_file in mist_files:
        isochrones.append(read_mist.ISO(iso_file, verbose=False))

    for isochrone, metallicity in zip(isochrones, metallicities):
        print metallicity, isochrone.abun['[Fe/H]']
        if metallicity > -0.7:
            age = 3e9
        else:
            age = 13.5e9
        i = isochrone.age_index(age)
        j = ((isochrone.isos[i]['phase'] != 3) &
             (isochrone.isos[i]['phase'] != 4) &
             (isochrone.isos[i]['phase'] != 5) &
             (isochrone.isos[i]['phase'] != 6))

        teff = isochrone.isos[i]['log_Teff'][j]
        logg = isochrone.isos[i]['log_g'][j]

        c3kout = 'MIST_{}_c3k_ew.txt'.format(metallicity)
        spiout = 'MIST_{}_psi_ew.txt'.format(metallicity)

        c3k_stars, spi_stars = [], []
        for i, (t, g) in enumerate(zip(teff, logg)):

            teff, logg = 10**t, g
            if teff <= 2800.:
                continue
            if logg < (-0.5):
                logg = (-0.5)

            # For C3K
            wave, flux, unc = sps.get_star_spectrum(Z=(10**(metallicity))*0.0190,
                        logt=t, logg=logg)
            c3k_stars.append(Star('mist', 10**t, logg, flux, wave))

            #'''
            #  For spi

            # Giants
            if (teff >= 2500. and teff <= 3500. and logg <= 4.0 and logg >= -0.5):
                func = spi['Cool Giants']
                flux = func.get_star_spectrum(logt=np.log10(teff),
                        logg=logg, feh=metallicity)
            elif (teff >= 4500. and teff <= 5500. and logg <= 4.0 and logg >= -0.5):
                func = spi['Warm Giants']
                flux = func.get_star_spectrum(logt=np.log10(teff),
                        logg=logg, feh=metallicity)
            elif (teff >= 5500. and teff < 6500. and logg <= 4.0 and logg >= -0.5):
                func1 = spi['Warm Giants']
                func2 = spi['Hot Stars']
                flux1 = func1.get_star_spectrum(logt=np.log10(teff),
                        logg=logg, feh=metallicity)
                flux2 = func2.get_star_spectrum(logt=np.log10(teff),
                        logg=logg, feh=metallicity)

                t_index = (np.abs(gh_teff_overlap - teff)).argmin()
                weight = gh_weights[t_index]
                flux = (flux1*weight + flux2*(1-weight))
            elif (teff >= 3500. and teff < 4500. and logg <= 4.0 and logg >= -0.5):
                func1 = spi['Cool Giants']
                func2 = spi['Warm Giants']
                flux1 = func1.get_star_spectrum(logt=np.log10(teff),
                        logg=logg, feh=metallicity)
                flux2 = func2.get_star_spectrum(logt=np.log10(teff),
                        logg=logg, feh=metallicity)

                t_index = (np.abs(gc_teff_overlap - teff)).argmin()
                weight = gc_weights[t_index]
                flux = (flux1*weight + flux2*(1-weight))


            # Dwarfs
            elif (teff >= 5500. and teff < 6000. and logg > 4.0):
                func = spi['Warm Dwarfs']
                flux = func.get_star_spectrum(logt=np.log10(teff),
                        logg=logg, feh=metallicity)
            elif (teff >= 2500. and teff <= 3000. and logg > 4.0):
                func = spi['Cool Dwarfs']
                flux = func.get_star_spectrum(logt=np.log10(teff),
                        logg=logg, feh=metallicity)
            elif (teff >= 3000. and teff <= 5500. and logg > 4.0):
                func1 = spi['Cool Dwarfs']
                func2 = spi['Warm Dwarfs']
                flux1 = func1.get_star_spectrum(logt=np.log10(teff),
                        logg=logg, feh=metallicity)
                flux2 = func2.get_star_spectrum(logt=np.log10(teff),
                        logg=logg, feh=metallicity)

                t_index = (np.abs(d_teff_overlap - teff)).argmin()
                weight = d_weights[t_index]
                flux = (flux1*weight + flux2*(1-weight))

            # Hot stars, have to split this up bcuz of warm stars
            elif (teff >= 6500. and teff <= 12e3 and logg <= 4.0 and logg >= -0.5):
                func = spi['Hot Stars']
                flux = func.get_star_spectrum(logt=np.log10(teff),
                        logg=logg, feh=metallicity)
            elif (teff >= 6000. and teff <= 12e3 and logg > 4.0):
                func = spi['Hot Stars']
                flux = func.get_star_spectrum(logt=np.log10(teff),
                        logg=logg, feh=metallicity)
            else:
                error = ('Parameter out of bounds:'
                         'teff = {0},  logg {1}')
                raise ValueError(error.format(teff, logg))

            wave = spi['Hot Stars'].wavelengths*1e4

            plt.plot(wave, flux)
            plt.savefig('SPI_spectra/{0}_{1}_{2}.png'.format(metallicity, teff, logg))
            plt.cla()
            plt.clf()

            #spi_stars.append(Star('mist', teff,
            #                 logg, flux, wave))
            #'''

        #write_equiv_widths(linelist, c3k_stars, c3kout)
        #write_equiv_widths(linelist, spi_stars, spiout)

def coefficient_test(linelist):
    """ Testing the from_coefficients function in SPI_Utils
    """
    import get_spi_spectrum

    isochrone_path = ('../MistIsochrones/'
                      'IRTF_Metallicities/')
    mist_files = glob.glob(isochrone_path + 'MIST*')
    metallicities = [0.0]
    #metallicities = [0.0, 0.3, -0.5, -1.0, -1.5]

    isochrones = []
    for iso_file in mist_files:
        isochrones.append(read_mist.ISO(iso_file, verbose=False))

    for isochrone, metallicity in zip(isochrones, metallicities):
        print metallicity, isochrone.abun['[Fe/H]']
        if metallicity > -0.7:
            age = 3e9
        else:
            age = 13.5e9
        i = isochrone.age_index(age)
        j = ((isochrone.isos[i]['phase'] != 3) &
             (isochrone.isos[i]['phase'] != 4) &
             (isochrone.isos[i]['phase'] != 5) &
             (isochrone.isos[i]['phase'] != 6))

        teff = isochrone.isos[i]['log_Teff'][j]
        logg = isochrone.isos[i]['log_g'][j]

        spiout = 'MIST_{}_coeff_test_ew.txt'.format(metallicity)

        spi_stars = []
        for i, (t, g) in enumerate(zip(teff, logg)):
            s = get_spi_spectrum.from_coefficients(t, g, metallicity)

            plt.plot(s['wave'], s['flux'])
            plt.savefig('SPI_spectra/Coeffs_Test/{0}_{1}_{2}.png'.format(metallicity, 10**t, g))
            plt.cla()
            plt.clf()

            #spi_stars.append(Star('mist', t, g, s['flux'], s['wave']))

        #write_equiv_widths(linelist, spi_stars, spiout)


def franken_isochrone(linelist):
    from psi_stuff import spi
    sps = StarBasis('../StellarLibraries/ckc_models/ckc14_logl.flat.h5',
                    n_neighbors=1, verbose=True)

    iso_file = ('/Users/alexa/NonSolarModels/'
                'TheModels/OldIsochrones/stitched_iso_t13.5.dat')
    isochrone = Table.read(iso_file, format='ascii.commented_header')
    i =  (isochrone['Phase'] != 4)
    teff, logg = isochrone['logt'][i], isochrone['logg'][i]


    c3kout = 'test_new_franken_iso_c3k_ews.txt'
    spiout = 'test_new_franken_iso_psi_ews.txt'

    c3k_stars, spi_stars = [], []
    for t, g in zip(teff, logg):
        if 10**t <= 3000.:
            continue

        # For C3K
        wave, flux, unc = sps.get_star_spectrum(Z=0.019,
                    logt=t, logg=g)
        c3k_stars.append(Star('franken_iso', 10**t,
                     g, flux, wave))

        #  For spi
        teff, logg = 10**t, g
        if teff < 2800.:
            teff = 2800
        if logg < (-0.5):
            logg = (-0.5)

        if (teff >= 2500. and teff <= 4000. and logg <= 3.5 and logg >= -0.5):
            func = spi['Cool Giants']
        elif (teff >= 2500. and teff <= 4000. and logg > 3.5):
            func = spi['Cool Dwarfs']
        elif (teff >= 4000. and teff < 6000. and logg <= 3.5 and logg >= -0.5):
            func = spi['Warm Giants']
        elif (teff >= 4000. and teff < 6000. and logg > 3.5):
            func = spi['Warm Dwarfs']
        elif (teff >= 6000. and teff <= 12e3 and logg < 5.0):
            func = spi['Hot Stars']
        else:
            error = ('Parameter out of bounds:'
                     'teff = {0},  logg {1}')
            raise ValueError(error.format(teff, logg))

        flux = func.get_star_spectrum(logt=np.log10(teff),
                logg=logg, feh=0.0)
        flux = stats.sigma_clip(flux)

        spi_stars.append(Star('franken_iso', 10**t,
                     g, flux, spi['Hot Stars'].wavelengths*1e4))
    #write_equiv_widths(linelist, c3k_stars, c3kout)
    write_equiv_widths(linelist, spi_stars, spiout)

def check_w_charlie(linelist):

    spec = Table.read('vcj_cc4_Zp0.00_t13.5.ssp',
                       format='ascii.commented_header')

    #spec = Table.read('m0.00_13.0.ssp',
    #                   format='ascii.commented_header')

    model = Star(None, None, None, spec["salp,"], spec["lam,"])
    line_names, line_strengths, upper, lower = get_equiv_widths(linelist, model)

    for name, val in zip(line_names, line_strengths):
        print name, val


def ssp_irtf(linelist):
    path = ('../SPS_Plots/SSPModels/NewModels')
    models = glob.glob('{}/*.ssp'.format(path))

    fname = 'ssp_ews_vcj_v4.txt'
    with open(fname, 'w+') as f:
        for i, model in enumerate(models):
            tmp = model.strip(path).strip('.ssp')
            met, age, ver = tmp.split('_')
            age = age.replace('t','')
            spec = Table.read(model, format='ascii.commented_header')
            # Degrade resolution of spectrum
            gauss = Gaussian1DKernel(stddev=4.8)
            dspec = convolve(spec['x=2.35'], gauss)
            for j, col in enumerate(spec.colnames):
                if col == 'lambda (vac)':
                    continue
                dspec = spec[col]
                model = Star(None, None, None, dspec, spec['lambda (vac)']*1e4)
                line_names, line_strengths, upper, lower = get_equiv_widths(linelist, model)

                if (i == 0 and j == 1):
                    f.write("# Age [Fe/H] IMF ")
                    for name in line_names:
                        f.write("{:15}".format(name))
                    f.write("\n")
                f.write("{0:1.4} {1:5.5} {2:10}".format(age, met.replace('m', '-'), col))
                for value in line_strengths:
                    f.write("{0:15.5f}".format(value))
                f.write("\n")

    subprocess.call(["mv", fname,
                     "/Users/alexa/NonSolarModels/TheModels/LickIndices"])

def ssp_cvd(linelist):
    path = '/Users/alexa/Dropbox (ConroyAstro)/alf/empirical SSPs'
    models = glob.glob('{}/*0.ssp'.format(path))
    models.append('{}/CvD_t13.5.ssp'.format(path))
    fname = 'ssp_ews_cvd_v4.txt'

    with open(fname, 'w+') as f:
        for i, model in enumerate(models):
            age = model.strip(path).strip('.ssp').strip('CvD_t')
            spec = np.loadtxt(model)
            gauss = Gaussian1DKernel(stddev=4.8)
            dspec = convolve(spec[:,3], gauss)

            model = Star(None, None, None, dspec, spec[:,0])
            line_names, line_strengths, upper, lower = get_equiv_widths(linelist, model)

            if (i == 0):
                f.write("# Age ")
                for name in line_names:
                    f.write("{:15}".format(name))
                f.write("\n")
            f.write("{0:1.4}".format(age))
            for value in line_strengths:
                f.write("{0:15.5f}".format(value))
            f.write("\n")

    subprocess.call(["mv", fname,
                     "/Users/alexa/NonSolarModels/TheModels/LickIndices"])

def mock_data(linelist):
    path = '/Users/alexa/Documents/AlfData/MockData/KCWI/Mg_SN_Test/'
    mocks = glob.glob('{}*.dat'.format(path))
    fname = 'kcwi_sn_mg_test.txt'

    with open(fname, 'w+') as f:
        for i, mock in enumerate(mocks):
            sn = int(mock.split('sn')[1][0:4])
            spec = np.loadtxt(mock)

            model = Star(None, None, None, spec[:,1], spec[:,0])
            model.add_uncertainty(spec[:,2])
            line_names, line_strengths, upper, lower = get_equiv_widths(linelist, model, error=True)

            if (i == 0):
                f.write("# SN ")
                for name in line_names:
                    f.write("{0:15} {1:15} {2:15}".format(name, name+'_u', name+'_l'))
                f.write("\n")
            f.write("{0:1}".format(sn))
            for val1, val2, val3 in zip(line_strengths, upper, lower):
                f.write("{0:15.5f}{1:15.5f}{2:15.5f}".format(val1, val2, val3))
            f.write("\n")

    subprocess.call(["mv", fname, path])

if __name__ == '__main__':

    #linelist = 'ziel_line_list.txt'
    #galaxies(linelist)
    #linelist = 'kcwi_test_line_list.txt'
    #mock_data(linelist)
    #sys.exit()

    linelist = 'full_line_list.txt'
    #old_irtf_lib(linelist)
    #franken_isochrone(linelist)

    #mist_isochrone(linelist)
    coefficient_test(linelist)
    #new_irtf_lib(linelist, model=False)

    #linelist = 'lick_indices_line_list.txt'
    #check_w_charlie(linelist)
    #ssp_irtf(linelist)
    #ssp_cvd(linelist)
