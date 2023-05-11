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
    import cartopy.feature as cseature

    # Load a high-resolution (1:10m) map of country borders
    Borders = cseature.NaturalEarthFeature(
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
            alpha=0.8, zorder=9)

    # Add coastlines to the axes
    ax.coastlines(resolution='110m', color='k', alpha=0.78, lw=0.6, zorder=10)

    # Add country borders to the axes
    ax.add_feature(Borders, edgecolor='gray', facecolor='None',
                   alpha=0.8, lw=0.6, zorder=11)

    return ax


def define_grid_fig(num_rows, num_columns,
                    horiz_spacing=0.015, vert_spacing=0.05, **kwargs):
    """
    Calculate the coordinates and dimensions of the subplots in a grid figure.

    Parameters
    ----------
    num_rows : int
        The number of rows in the grid.
    num_columns : int
        The number of columns in the grid.
    horiz_spacing : float, optional
        The horizontal spacing between subplots, by default 0.015.
    vert_spacing : float, optional
        The vertical spacing between subplots, by default 0.05.
    **kwargs : dict, optional
        Additional keyword arguments for customizing the borders of the grid. 
        These can include 'left_border', 'right_border', 'top_border',
        and 'bottom_border'. 
        If not provided, default values are 0.01 for 'left_border'
        and 'right_border' and 0.03 for 'top_border' and 'bottom_border'.

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
    x_corner = kwargs.get(
        'x_corner', lambda x: left_border + (x) * (x_fig + horiz_spacing))

    # Calculate the width of each subplot
    x_fig = (1 - (left_border + right_border +
             (num_columns - 1) * horiz_spacing)) / num_columns

    # Calculate the x-coordinates of the lower-left corner of each subplot
    x_coords = [x_corner(i) for i in range(num_columns)]

    # Set the top and bottom borders, and vertical spacing between subplots
    top_border = kwargs.get('top_border', 0.03)
    bottom_border = kwargs.get('bottom_border', 0.03)

    # Calculate the height of each subplot
    y_fig = (1 - (top_border + bottom_border +
             (num_rows - 1) * vert_spacing)) / num_rows

    # Calculate the y-coordinates of the lower-left corner of each subplot
    y_coords = np.flip([bottom_border + i * (y_fig + vert_spacing)
                        for i in range(num_rows)])

    return x_coords, y_coords, x_fig, y_fig


def add_colorbar(fig, cs, label, orientation, grid_prop,
                 cbar_factor=0.8, cbar_width=0.025, **kwargs):
    """
    Add a colorbar to a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to which the colorbar will be added.
    cs : QuadContourSet
        The contour plot for which the colorbar will be created.
    label : str
        The label for the colorbar.
    orientation : str
        The orientation of the colorbar, either 'horizontal' or 'vertical'.
    grid_prop : tuple
        Tuple containing the x-coordinates, y-coordinates, width,
        and height of each subplot.
    cbar_factor : float, optional
        The scaling factor for the colorbar, by default 0.8.
        Determines the length of the colorbar relative to the plot.
    cbar_width : float, optional
        The width of the colorbar, by default 0.025.
    **kwargs : dict, optional
        Additional keyword arguments for customizing the position of the
        colorbar. Can include 'y_coord_cbar' and 'x_coord_cbar' for vertical
        and horizontal colorbars respectively.

    Raises
    ------
    ValueError
        If the orientation is neither 'horizontal' nor 'vertical'.
    """

    # Unpack the properties of the grid
    (x_coords, y_coords, x_fig, y_fig) = grid_prop

    # Get the y-coordinate for the colorbar, default to -0.1 if not specified
    y_coord_cbar = kwargs.get('y_coord_cbar', -0.1)

    # Get the x-coordinate for the colorbar, default to 1 if not specified
    x_coord_cbar = kwargs.get('x_coord_cbar', 1)

    # Check the orientation of the colorbar
    if orientation == 'horizontal':
        # Calculate the axes of the colorbar for a horizontal orientation
        cbaxes = fig.add_axes([
            x_coords[0] + (1-cbar_factor)*(x_coords[-1]+x_fig-x_coords[0])/2,
            y_coord_cbar,
            (x_coords[-1]+x_fig-x_coords[0])*cbar_factor,
            cbar_width])

        # Add a horizontal colorbar to the figure
        fig.colorbar(cs, cax=cbaxes, orientation='horizontal', label=label)

    elif orientation == 'vertical':
        # Calculate the axes of the colorbar for a vertical orientation
        cbaxes = fig.add_axes([x_coord_cbar,
                               y_coords[-1] + (1-cbar_factor) *
                               (y_coords[0]+y_fig-y_coords[-1])/2,
                               cbar_width,
                               (y_coords[0]+y_fig-y_coords[-1])*cbar_factor])

        # Add a vertical colorbar to the figure
        fig.colorbar(cs, cax=cbaxes, label=label)

    else:
        # Raise an error if the orientation is not recognized
        raise ValueError(
            "Invalid orientation. Choose either 'horizontal' or 'vertical'.")


if __name__ == '__main__':

    # Open the netCDF dataset
    ds = xr.open_dataset('dummy_data/air.2m.gauss.2022.nc')

    # Extract the temperature values (converting from Kelvin to Celsius)
    var_values = ds['air'].values[:, 0, :, :]-273.15
    # Extract the time values and convert to datetime
    time = pd.to_datetime(ds['time'].values)
    # Extract latitude and longitude values
    lat = ds['lat'].values
    lon = ds['lon'].values

    # Define the map projection (PlateCarree) and set the image extent
    proj = ccrs.PlateCarree(central_longitude=0)
    img_extent = (-115, -30, -10, 30)

    # Define the grid size (number of rows and columns)
    num_rows = 3
    num_columns = 3

    # Use the function to calculate properties of the grid
    grid_prop = x_coords, y_coords, x_fig, y_fig = define_grid_fig(
        num_rows, num_columns)

    # Define font properties for axis labels and title
    font_prop = {'fontsize': 12, 'fontweight': 'semibold', 'color': '#434343'}
    font_prop_title = {'fontsize': 14,
                       'fontweight': 'semibold', 'color': '#434343'}

    # Create a figure with a specified size
    fig = plt.figure(figsize=(8, 6))

    # Initialize the index for selecting time slices of the temperature data
    idx = 0
    # Define the contour levels for the temperature plot
    levels = np.linspace(6, 32, 18)

    # Define the colormap for the plot
    cmap = sns.color_palette("Spectral_r", as_cmap=True)

    # Loop through each row and column to create a grid of subplots
    for fi in range(num_rows):
        for ci in range(num_columns):
            # Add axes to the figure with the calculated properties
            ax = fig.add_axes([x_coords[ci], y_coords[fi],
                               x_fig, y_fig],
                              projection=proj)
            # Add geographic features to the plot
            ax = continentes_lon_lat(ax)

            # Set the image extent and aspect ratio of the plot
            ax.set_extent(img_extent, proj)
            ax.set_aspect('auto')

            # Remove y-axis labels for subplots that are not in the first column
            if ci > 0:
                ax.set_yticklabels([])

            # Remove x-axis labels for subplots that are not in the last row
            if fi < (num_rows - 1):
                ax.set_xticklabels([])

            # Plot the temperature data for the current time slice
            cs = ax.contourf(lon, lat, var_values[idx, :, :], levels,
                             cmap=cmap, extend='both')

            # Add a title to each subplot
            ax.set_title(f"{time[idx].strftime('%Y-%b-%d')}",
                         fontdict=font_prop_title)

            # Increment the index to move to the next time slice
            idx += 1

    # Define the orientation and label of the colorbar
    orientation = 'horizontal'
    label = 'Temperature [°C]'

    # Add a horizontal colorbar to the figure
    add_colorbar(fig=fig, cs=cs, label=label,
                 orientation=orientation,
                 grid_prop=grid_prop,
                 cbar_factor=0.8,
                 cbar_width=0.025)

    # Add a vertical colorbar to the figure (optional)
    add_colorbar(fig, cs, label, 'vertical', grid_prop,
                 cbar_factor=0.8, cbar_width=0.025)

    # Show the figure with all its subplots and colorbars
    plt.show()


# %%
