/* references for navbar: 
 * https://stackoverflow.com/questions/69401316/when-i-scroll-down-the-window-it-is-not-removing-or-adding-the-active-class
 * https://stackoverflow.com/questions/23706003/changing-nav-bar-color-after-scrolling
 * https://stackoverflow.com/questions/70688456/change-navbar-colour-on-scroll-in-bootstrap-5
 * https://developer.mozilla.org/en-US/docs/Web/API/Window/scrollY
 */
const header = document.querySelector('.navbar');

window.onscroll = function() {
    if(window.scrollY > 250) {
        header.classList.add('navbar');
    }
    else {
        header.classList.remove('navbar');
    }
}


/* references for multipage click: 
 * https://stackoverflow.com/questions/7064998/how-to-make-a-link-open-multiple-pages-when-clicked
 * https://stackoverflow.com/questions/5566970/how-can-i-open-two-pages-from-a-single-click-without-using-javascript
 */
var BLSLinks = [
  "https://www.bls.gov/careeroutlook/2022/data-on-display/education-pays.htm",
  "https://www.bls.gov/cps/cpsaat11.htm",
  "https://www.bls.gov/oes/current/oes_nat.htm#00-0000",
  "https://www.bls.gov/cps/cpsaat23.htm"
]
var NCESLinks = [
	"https://nces.ed.gov/programs/digest/d16/tables/dt16_318.30.asp",
  "https://nces.ed.gov/programs/digest/d16/tables/dt16_322.40.asp",
  "https://nces.ed.gov/programs/digest/d16/tables/dt16_322.50.asp"
]
var CBLinks = [
  "https://www.census.gov/data/tables/time-series/demo/educational-attainment/cps-historical-time-series.html",
  "https://www.statista.com/statistics/184272/educational-attainment-of-college-diploma-or-higher-by-gender/"
]
var CAWPLinks = [
  "https://cawp.rutgers.edu/facts/current-numbers/women-elective-office-2022",
  "https://www.pewresearch.org/social-trends/fact-sheet/the-data-on-women-leaders/ "
]

function openMultipleLinks(links) {
	for (var i = 0; i < links.length; i ++) {
  	window.open(links[i]);
  }	
}