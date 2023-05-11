# %%
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid")
sns.set_context('notebook', font_scale=1.2)


def continentes_lon_lat(ax, lon_step=30, lat_step=15):
    """
    Add continents, coastlines, gridlines, and tick labels to a Cartopy axes.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The Cartopy axes to modify.
    lon_step : int, optional
        The step size for longitude gridlines and tick labels, by default 30.
    lat_step : int, optional
        The step size for latitude gridlines and tick labels, by default 15.

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxesSubplot
        The modified Cartopy axes.
    """
    import numpy as np
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import cartopy.feature as cfeature

    # Load a high-resolution (1:10m) map of country borders
    Borders = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale='110m',
        facecolor='none'
    )

    # Set the tick locations and labels for the axes
    ax.set_xticks(np.arange(-180, 180, lon_step), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, lat_step), crs=ccrs.PlateCarree())
    ax.tick_params(axis='both', which='major', labelsize=14, color="#434343")
    lon_formatter = LongitudeFormatter(zero_direction_label=True,
                                       number_format='.0f')
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_axisbelow(False)

    # Add gridlines to the axes
    ax.grid(which='major', linestyle='--', linewidth='0.6', color='gray',
            alpha=0.8)

    # Add coastlines to the axes
    ax.coastlines(resolution='110m', color='k', alpha=0.78, lw=0.6)

    # Add country borders to the axes
    ax.add_feature(Borders, edgecolor='gray', facecolor='None',
                   alpha=0.8, lw=0.6)

    return ax


def define_grid_fig(num_fil, num_col, **kwargs):
    """
    Calculate the coordinates and dimensions of the subplots in a grid figure.

    Parameters
    ----------
    num_fil : int
        The number of rows in the grid.
    num_col : int
        The number of columns in the grid.
    **kwargs : dict, optional
        Keyword arguments for customizing the borders and spacing of the grid.

    Returns
    -------
    x_coords : list
        List of x-coordinates of the lower-left corner of each subplot.
    y_coords : list
        List of y-coordinates of the lower-left corner of each subplot.
    x_fig : float
        Width of each subplot.
    y_fig : float
        Height of each subplot.
    """

    # Set the left and right borders, and horizontal spacing between subplots
    left_border = kwargs.get('left_border', 0.01)
    right_border = kwargs.get('right_border', 0.03)
    horiz_spacing = kwargs.get('horiz_spacing', 0.015)
    x_corner = kwargs.get(
        'x_corner', lambda x: left_border + (x) * (x_fig + horiz_spacing))

    # Calculate the width of each subplot
    x_fig = (1 - (left_border + right_border +
             (num_col - 1) * horiz_spacing)) / num_col

    # Calculate the x-coordinates of the lower-left corner of each subplot
    x_coords = [x_corner(i) for i in range(num_col)]

    # Set the top and bottom borders, and vertical spacing between subplots
    top_border = kwargs.get('top_border', 0.03)
    bottom_border = kwargs.get('bottom_border', 0.03)
    vert_spacing = kwargs.get('vert_spacing', 0.02)

    # Calculate the height of each subplot
    y_fig = (1 - (top_border + bottom_border +
             (num_fil - 1) * vert_spacing)) / num_fil

    # Calculate the y-coordinates of the lower-left corner of each subplot
    y_coords = np.flip([bottom_border + i * (y_fig + vert_spacing)
                        for i in range(num_fil)])

    return x_coords, y_coords, x_fig, y_fig


def add_colorbar(fig, cf, label, orientation, x_coords, y_coords,
                 x_fig, y_fig, cbar_factor=0.8, cbar_width=0.025):
    """
    Add a colorbar to a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to which the colorbar will be added.
    cf : QuadContourSet
        The contour plot for which the colorbar will be created.
    orientation : str
        The orientation of the colorbar, either 'horizontal' or 'vertical'.
    x_coords : list
        List of x-coordinates of the lower-left corner of each subplot.
    y_coords : list
        List of y-coordinates of the lower-left corner of each subplot.
    x_fig : float
        Width of each subplot.
    y_fig : float
        Height of each subplot.
    cbar_factor : float, optional
        The scaling factor for the colorbar, by default 0.8.
    label : str, optional
        The label for the colorbar, by default 'Temperature [°C]'.
    """
    y_corner_cbar = -0.1
    
    if orientation == 'horizontal':
        cbaxes = fig.add_axes([
            x_coords[0] + (1-cbar_factor)*(x_coords[-1]+x_fig-x_coords[0])/2,
            y_corner_cbar,
            (x_coords[-1]+x_fig-x_coords[0])*cbar_factor,
            cbar_width])

        fig.colorbar(cf, cax=cbaxes, orientation='horizontal', label=label)

    elif orientation == 'vertical':
        cbaxes = fig.add_axes([1,
                               y_coords[-1] + (1-cbar_factor) *
                               (y_coords[0]+y_fig-y_coords[-1])/2,
                               cbar_width,
                               (y_coords[0]+y_fig-y_coords[-1])*cbar_factor])
        fig.colorbar(cf, cax=cbaxes, label=label)

    else:
        raise ValueError(
            "Invalid orientation. Choose either 'horizontal' or 'vertical'.")


if __name__ == '__main__':

    ds = xr.open_dataset('dummy_data/air.2m.gauss.2022.nc')

    var_values = ds['air'].values[:, 0, :, :]-273.15
    time = pd.to_datetime(ds['time'].values)
    lat = ds['lat'].values
    lon = ds['lon'].values

    # Define the map projection and image extent
    proj = ccrs.PlateCarree(central_longitude=0)
    img_extent = (-115, -30, -10, 30)

    # Define the number of rows and columns for the grid of subplots
    num_fil = 3
    num_col = 3

    # Calculate the coordinates, width, and height of each subplot
    x_coords, y_coords, x_fig, y_fig = define_grid_fig(num_fil, num_col,
                                                       horiz_spacing=0.02,
                                                       vert_spacing=0.05)

    # Define font properties for axis labels and title
    font_prop = {'fontsize': 12, 'fontweight': 'semibold', 'color': '#434343'}
    font_prop_title = {'fontsize': 14,
                       'fontweight': 'semibold', 'color': '#434343'}

    # Create the figure
    fig = plt.figure(figsize=(8, 6))

    # Initialize the index for selecting time slices of the temperature data
    idx = 0
    # Define the contour levels for the temperature plot
    levels = np.linspace(6, 32, 18)

    cmap = sns.color_palette("Spectral_r", as_cmap=True)

    # Loop through columns and rows to create subplots
    for fi in range(num_fil):
        for ci in range(num_col):
            # Add axes to the figure with given coordinates, width, and height
            ax = fig.add_axes([x_coords[ci], y_coords[fi],
                               x_fig, y_fig],
                              projection=proj)
            # Add continents, coastlines, and gridlines to the subplot
            ax = continentes_lon_lat(ax)

            # Set the extent and aspect ratio of the subplot
            ax.set_extent(img_extent, proj)
            ax.set_aspect('auto')

            # Remove y-axis labels for subplots that are not in the first
            # column
            if ci > 0:
                ax.set_yticklabels([])

            # Remove x-axis labels for subplots that are not in the last row
            if fi < (num_fil - 1):
                ax.set_xticklabels([])

            # Plot the temperature data for the current time slice
            cf = ax.contourf(lon, lat, var_values[idx, :, :], levels,
                             cmap=cmap, extend='both')

            # Add title to the current subplot
            ax.set_title(f"{time[idx].strftime('%Y-%b-%d')}",
                         fontdict=font_prop_title)

            # Increment the index to select the next time slice
            idx += 1

    orientation = 'horizontal'
    label = 'Temperature [°C]'

    add_colorbar(fig, cf, label, 'horizontal', x_coords, y_coords, x_fig, y_fig,
                 cbar_factor=0.8, cbar_width=0.025)
    
    
    add_colorbar(fig, cf, label, 'vertical', x_coords, y_coords, x_fig, y_fig,
                 cbar_factor=0.8, cbar_width=0.025)

# %%
